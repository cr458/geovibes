"""Data access and configuration helpers for the GeoVibes UI."""

from __future__ import annotations

import csv
import hashlib
import os
import pathlib
import queue
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import faiss
import geopandas as gpd
import numpy as np

from geovibes.database.faiss_cache import FaissCache
from geovibes.ee_tools import initialize_ee_with_credentials
from geovibes.ui_config import BasemapConfig, DatabaseConstants, GeoVibesConfig

from .utils import (
    get_database_centroid,
    infer_tile_spec_from_name,
    list_databases_in_directory,
    log_to_file,
    prepare_ids_for_query,
)


class DataManager:
    """Encapsulates configuration, database, and FAISS operations."""

    @staticmethod
    def is_remote_url(path: str) -> bool:
        """Check if a path is a remote URL (S3 or GCS).

        Args:
            path: File path or URL to check.

        Returns:
            True if the path is a remote URL (s3:// or gs://), False otherwise.
        """
        if not path:
            return False
        return path.startswith("s3://") or path.startswith("gs://")

    @staticmethod
    def _normalize_query_mode(raw: Optional[str]) -> str:
        mode = (raw or "in_list").strip().lower()
        if mode in {"values", "values_join"}:
            return "values_join"
        return "in_list"

    @staticmethod
    def _normalize_batch_scheduler(raw: Optional[str]) -> str:
        mode = (raw or "id_ascending").strip().lower()
        if mode in {"none", "as_is"}:
            return "as_is"
        if mode in {"id_ascending", "ascending", "asc"}:
            return "id_ascending"
        if mode in {"id_descending", "descending", "desc"}:
            return "id_descending"
        return "id_ascending"

    @staticmethod
    def _coerce_id_list(ids: List[object]) -> List[object]:
        coerced: List[object] = []
        for value in ids:
            try:
                coerced.append(int(value))  # type: ignore[arg-type]
            except Exception:
                coerced.append(str(value))
        return coerced

    def __init__(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        duckdb_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        geometry_cache_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        config: Optional[Dict] = None,
        enable_ee: Optional[bool] = None,
        config_path: Optional[str] = None,
        duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None,
        baselayer_url: Optional[str] = None,
        disable_ee: bool = False,
        verbose: bool = False,
        include_remote: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        self.verbose = verbose
        self.include_remote = include_remote
        self.baselayer_url = baselayer_url or BasemapConfig.BASEMAP_TILES["MAPTILER"]
        self.duckdb_path = duckdb_path
        self.faiss_path = faiss_path
        self.geometry_cache_path = geometry_cache_path
        self.duckdb_directory = duckdb_directory

        if "enable_ee" in unused_kwargs and self.verbose:
            print(
                "ℹ️ Pass enable_ee via config or GEOVIBES_ENABLE_EE environment variable."
            )

        # Configuration and Earth Engine toggles
        self.config = self._load_config(
            start_date=start_date,
            end_date=end_date,
            config=config,
            config_path=config_path,
            enable_ee_override=enable_ee,
        )

        self.ee_available = False
        env_enable = os.getenv("GEOVIBES_ENABLE_EE")
        env_opt_in = bool(
            env_enable and env_enable.strip().lower() in {"1", "true", "yes", "on"}
        )
        ee_opt_in = (self.config.enable_ee or env_opt_in) and not disable_ee
        if ee_opt_in:
            self.ee_available = initialize_ee_with_credentials(verbose=self.verbose)
        elif self.verbose and not disable_ee:
            print(
                "ℹ️ Earth Engine basemaps disabled (enable via config or GEOVIBES_ENABLE_EE)"
            )

        self.geometries_dir = self._resolve_geometries_directory()
        self.local_database_directory = self._resolve_local_database_directory()
        self.manifest_entries: List[Dict[str, str]] = []
        self.id_column_candidates: List[str] = ["id"]
        self.external_id_column: str = "id"

        self.available_databases = self._discover_databases()
        for entry in self.available_databases:
            entry.setdefault("tile_spec", infer_tile_spec_from_name(entry["db_path"]))
        if not self.available_databases:
            raise FileNotFoundError(
                "No downloaded models found. Provide duckdb_path/duckdb_directory or run prep_data.py."
            )

        self.available_databases = sorted(
            self.available_databases,
            key=lambda entry: entry.get(
                "display_name", os.path.basename(entry["db_path"])
            ),
        )
        self.database_info_by_path = {
            entry["db_path"]: entry for entry in self.available_databases
        }

        # Initialize connection state variables
        self.current_database_info: Optional[Dict] = None
        self.current_database_path: Optional[str] = None
        self.current_faiss_path: Optional[str] = None
        self.current_geometry_path: Optional[str] = None
        self.current_geometry_cache_path: Optional[str] = None
        self.tile_spec: Optional[Dict] = None
        self.effective_boundary_path: Optional[str] = None
        self.duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None
        self._owns_connection = False
        self.faiss_index: Optional[faiss.Index] = None
        self.embedding_dim: Optional[int] = None
        self.embedding_type: Optional[str] = None
        self._embedding_select_expr: str = os.getenv(
            "GEOVIBES_EMBEDDING_SELECT_EXPR", "embedding"
        )
        self._geometry_cache_local_path: Optional[pathlib.Path] = None
        self._geometry_cache_connection: Optional[duckdb.DuckDBPyConnection] = None
        self._row_group_cache_local_path: Optional[pathlib.Path] = None
        self._row_group_cache_connection: Optional[duckdb.DuckDBPyConnection] = None
        self._row_group_size: int = 122880
        self._background_pool_size: int = 0
        self._background_pool_created: int = 0
        self._background_pool_queue: Optional[
            queue.LifoQueue[duckdb.DuckDBPyConnection]
        ] = None
        self._background_pool_lock = threading.Lock()
        self._metadata_query_mode = self._normalize_query_mode(
            os.getenv("GEOVIBES_METADATA_QUERY_MODE", "values_join")
        )
        self._prefetch_query_mode = self._normalize_query_mode(
            os.getenv("GEOVIBES_PREFETCH_QUERY_MODE", "in_list")
        )
        self._prefetch_batch_scheduler = self._normalize_batch_scheduler(
            os.getenv("GEOVIBES_PREFETCH_BATCH_SCHEDULER", "id_ascending")
        )
        self.center_x: float = 0.0
        self.center_y: float = 0.0

        # Deferred loading: skip connection when include_remote=True
        # User will select a database from dropdown, then we connect
        if self.include_remote:
            if self.verbose:
                print("📋 Deferred loading enabled - select a database to connect")
            return

        # Immediate loading: connect to first database
        first_db = self.available_databases[0]
        self._connect_to_database_internal(first_db, duckdb_connection)

    # ------------------------------------------------------------------
    # Database connection
    # ------------------------------------------------------------------

    def connect_to_database(self, db_path: str) -> None:
        """Connect to a database by path. Used for deferred loading."""
        db_info = self.database_info_by_path.get(db_path)
        if not db_info:
            raise ValueError(f"Unknown database: {db_path}")
        self._connect_to_database_internal(db_info, duckdb_connection=None)

    def _connect_to_database_internal(
        self,
        db_info: Dict,
        duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        """Internal method to connect to a database."""
        if getattr(self, "_owns_connection", False) and getattr(
            self, "duckdb_connection", None
        ):
            self.duckdb_connection.close()

        self._close_auxiliary_connections()

        self.current_database_info = db_info
        self.current_database_path = db_info["db_path"]
        self.current_faiss_path = db_info.get("faiss_path")
        self.current_geometry_path = db_info.get("geometry_path")
        self.tile_spec = db_info.get("tile_spec")
        if not self.tile_spec:
            self.tile_spec = infer_tile_spec_from_name(self.current_database_path)
        self.effective_boundary_path = None

        # Manage DuckDB connection
        if duckdb_connection is None:
            self.duckdb_connection = self._connect_duckdb(self.current_database_path)
            self._owns_connection = True
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False

        self._apply_duckdb_settings(self.current_database_path)
        self._refresh_id_columns()

        # Load FAISS index
        if not self.current_faiss_path:
            raise ValueError("Could not find a FAISS index for the selected database.")
        if self.verbose:
            print(f"🧠 Loading FAISS index from: {self.current_faiss_path}")
        self.faiss_index = self._load_faiss_index(self.current_faiss_path)
        if self.verbose:
            print(f"✅ FAISS index loaded. Contains {self.faiss_index.ntotal} vectors.")

        # Detect embedding dimension
        self.embedding_dim = self._detect_embedding_dim()
        self.embedding_type = self._detect_embedding_type()
        self._embedding_select_expr = self._resolve_embedding_select_expr()

        # Load geometry cache for remote databases
        self.current_geometry_cache_path = db_info.get("geometry_cache_path")
        self._row_group_cache_local_path = None
        if self.current_geometry_cache_path and self.is_remote_url(
            self.current_database_path
        ):
            self._load_geometry_cache(self.current_geometry_cache_path)

        if self.is_remote_url(self.current_database_path):
            self._load_row_group_cache(self._row_group_size)

        # Warm up remote databases to preload row groups
        pool_precreate_thread: Optional[threading.Thread] = None
        if self.is_remote_url(self.current_database_path):
            pool_target = self.suggest_background_pool_size()
            pool_precreate_thread = self.start_background_pool_precreation(
                pool_target,
                n_workers=pool_target,
            )
            self._warm_up_remote_database()
            if pool_precreate_thread is not None and pool_precreate_thread.is_alive():
                pool_precreate_thread.join(timeout=10.0)

        # Derive map centering data
        self.effective_boundary_path, (self.center_y, self.center_x) = (
            self._setup_boundary_and_center()
        )

    def is_connected(self) -> bool:
        """Check if a database is currently connected."""
        return self.duckdb_connection is not None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_config(
        self,
        *,
        start_date: Optional[str],
        end_date: Optional[str],
        config: Optional[Dict],
        config_path: Optional[str],
        enable_ee_override: Optional[bool],
    ) -> GeoVibesConfig:
        if config_path is not None:
            cfg = GeoVibesConfig.from_file(config_path)
        elif config is not None:
            cfg = GeoVibesConfig.from_dict(config)
        else:
            cfg = GeoVibesConfig(
                start_date=start_date or "2024-01-01",
                end_date=end_date or "2025-01-01",
            )

        if enable_ee_override is not None:
            cfg.enable_ee = bool(enable_ee_override)

        if hasattr(cfg, "validate"):
            try:
                cfg.validate()
            except Exception as exc:  # pragma: no cover - defensive logging
                if self.verbose:
                    print(f"⚠️ Config validation skipped: {exc}")
        return cfg

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_root() -> pathlib.Path:
        # __file__ lives under geovibes/ui/, so ascend to repository root
        return pathlib.Path(__file__).resolve().parents[2]

    def _resolve_manifest_path(self) -> Optional[str]:
        override = os.getenv("GEOVIBES_MANIFEST_PATH")
        if override:
            candidate = pathlib.Path(override).expanduser()
            if candidate.exists():
                return str(candidate)

        manifest_path = self._project_root() / "manifest.csv"
        return str(manifest_path) if manifest_path.exists() else None

    def _resolve_geometries_directory(self) -> Optional[str]:
        override = os.getenv("GEOVIBES_GEOMETRIES_DIR")
        if override:
            return str(pathlib.Path(override).expanduser())
        return str(self._project_root() / "geometries")

    def _resolve_local_database_directory(self) -> str:
        if self.duckdb_directory:
            return str(pathlib.Path(self.duckdb_directory).expanduser())
        override = os.getenv("GEOVIBES_LOCAL_DB_DIR")
        if override:
            return str(pathlib.Path(override).expanduser())
        return str(self._project_root() / "local_databases")

    def _discover_databases(self) -> List[Dict[str, str]]:
        discovered: List[Dict[str, str]] = []

        db_path = self.duckdb_path or getattr(self.config, "duckdb_path", None)
        if db_path:
            faiss_path = self.faiss_path or self._infer_faiss_from_db(db_path)
            if not faiss_path:
                if self.verbose:
                    print(f"⚠️  Could not locate FAISS index for {db_path}. Skipping.")
            else:
                geometry_path = self._infer_geometry_from_db(db_path)
                if geometry_path is None:
                    geometry_path = getattr(self.config, "boundary_path", None)
                geometry_cache = (
                    self.geometry_cache_path
                    or self._infer_geometry_cache_from_db(db_path)
                )
                discovered.append(
                    {
                        "db_path": db_path,
                        "faiss_path": faiss_path,
                        "display_name": os.path.basename(db_path),
                        "geometry_path": geometry_path,
                        "geometry_cache_path": geometry_cache,
                    }
                )
                return discovered

        duckdb_directory = self.duckdb_directory or getattr(
            self.config, "duckdb_directory", None
        )
        if duckdb_directory:
            directory_entries = list_databases_in_directory(
                duckdb_directory, verbose=self.verbose
            )
            for entry in directory_entries:
                if not entry.get("faiss_path"):
                    if self.verbose:
                        print(
                            f"⚠️  Missing FAISS index for {entry['db_path']}. Skipping."
                        )
                    continue
                entry.setdefault("display_name", os.path.basename(entry["db_path"]))
                geometry_path = entry.get("geometry_path")
                if not geometry_path:
                    geometry_path = self._infer_geometry_from_db(entry["db_path"])
                if not geometry_path:
                    geometry_path = getattr(self.config, "boundary_path", None)
                entry["geometry_path"] = geometry_path
                discovered.append(entry)
            for entry in discovered:
                entry.setdefault(
                    "geometry_path", getattr(self.config, "boundary_path", None)
                )
            if discovered:
                return discovered

        # Fallback to manifest-driven discovery
        manifest_path = self._resolve_manifest_path()
        if manifest_path is None:
            return discovered

        self.manifest_entries = self._load_manifest_entries(manifest_path)
        if not self.manifest_entries:
            return discovered

        if self.verbose:
            print(
                f"📄 Loaded {len(self.manifest_entries)} manifest entries from {manifest_path}"
            )

        discovered.extend(
            self._discover_available_models(
                self.local_database_directory, self.manifest_entries
            )
        )

        # Tier 4: Remote databases from S3
        # Include if: (a) no local databases found, or (b) include_remote=True
        if not discovered or self.include_remote:
            try:
                remote_dbs = self.discover_remote_databases()
                if remote_dbs:
                    if self.verbose:
                        msg = "alongside local" if discovered else "no local found"
                        print(f"📡 Found {len(remote_dbs)} remote databases ({msg})")
                    discovered.extend(remote_dbs)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not list remote databases: {e}")

        return discovered

    def discover_remote_databases(self, base_url: str = None) -> List[Dict[str, Any]]:
        """Scan S3 for available remote databases in httpfs/ folders.

        Recursively scans from base_url to find all httpfs/ folders containing
        database files (metadata.db, faiss.index, geometry_cache.parquet).

        Args:
            base_url: S3 URL to start scanning from. Defaults to
                DatabaseConstants.DEFAULT_SEARCH_BASE_URL.

        Returns:
            List of database entries compatible with available_databases format.
            Each entry includes is_remote=True flag.
        """
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        base_url = base_url or DatabaseConstants.DEFAULT_SEARCH_BASE_URL

        # Parse S3 URL
        if not base_url.startswith("s3://"):
            return []

        # Extract bucket and prefix from URL
        url_parts = base_url[5:].split("/", 1)
        bucket = url_parts[0]
        prefix = url_parts[1] if len(url_parts) > 1 else ""

        if self.verbose:
            print(f"🔍 Scanning for remote databases at {base_url}")

        # Create S3 client (anonymous access for public bucket)
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )

        discovered = []

        # List all objects under the prefix to find httpfs/ folders
        paginator = s3.get_paginator("list_objects_v2")
        httpfs_folders = set()

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Look for httpfs/ in the path
                if "/httpfs/" in key:
                    # Extract the httpfs folder path (e.g., .../httpfs/model_name/)
                    httpfs_idx = key.index("/httpfs/")
                    after_httpfs = key[httpfs_idx + 8 :]  # Skip "/httpfs/"
                    if "/" in after_httpfs:
                        model_folder = after_httpfs.split("/")[0]
                        folder_path = key[: httpfs_idx + 8] + model_folder
                        httpfs_folders.add(folder_path)

        if self.verbose:
            print(f"   Found {len(httpfs_folders)} model folders")

        # For each httpfs folder, check for required files
        for folder_path in sorted(httpfs_folders):
            folder_prefix = folder_path + "/"

            # List files in this folder
            response = s3.list_objects_v2(
                Bucket=bucket, Prefix=folder_prefix, Delimiter="/"
            )
            files = {obj["Key"].split("/")[-1] for obj in response.get("Contents", [])}

            # Check for required files
            has_db = "metadata.db" in files
            has_faiss = "faiss.index" in files
            has_geometry = "geometry_cache.parquet" in files

            if not (has_db and has_faiss):
                if self.verbose:
                    print(f"   ⚠️  Skipping {folder_path} (missing required files)")
                continue

            # Build URLs
            model_name = folder_path.split("/")[-1]
            db_url = f"s3://{bucket}/{folder_prefix}metadata.db"
            faiss_url = f"s3://{bucket}/{folder_prefix}faiss.index"
            geometry_url = (
                f"s3://{bucket}/{folder_prefix}geometry_cache.parquet"
                if has_geometry
                else None
            )

            # Extract region and date range from path for display
            # Path format: .../USA/alabama/2024-01-01-2025-01-01/httpfs/model_name
            path_parts = folder_path.split("/")
            region = None
            date_range = None
            for i, part in enumerate(path_parts):
                if part == "httpfs" and i >= 2:
                    region = path_parts[i - 2]  # e.g., "alabama"
                    date_range = path_parts[i - 1]  # e.g., "2024-01-01-2025-01-01"
                    break

            # Build display name
            display_name = self._build_remote_display_name(
                model_name, region, date_range
            )

            # Parse tile spec from model name
            tile_spec = infer_tile_spec_from_name(model_name)

            # Build boundary URL from region
            boundary_url = None
            if region:
                boundary_url = f"s3://{DatabaseConstants.SOURCE_COOP_BUCKET}/geovibes/geometries/{region}.geojson"

            entry = {
                "db_path": db_url,
                "faiss_path": faiss_url,
                "display_name": display_name,
                "geometry_path": None,  # Remote DBs use geometry cache instead
                "geometry_cache_path": geometry_url,
                "boundary_url": boundary_url,
                "tile_spec": tile_spec,
                "is_remote": True,
                "region": region,
                "date_range": date_range,
            }
            discovered.append(entry)

            if self.verbose:
                print(f"   ✓ {display_name}")

        return discovered

    def _build_remote_display_name(
        self, model_name: str, region: str = None, date_range: str = None
    ) -> str:
        """Build a human-readable display name for a remote database."""
        # Extract model type from name
        # e.g., "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10"
        # -> "DINO ViT (32px, 16px, 10m)"

        # Parse tile spec for display
        tile_spec = infer_tile_spec_from_name(model_name)
        tile_info = ""
        if tile_spec:
            tile_info = f" ({tile_spec.get('tile_size_px', '?')}px)"

        # Simplify model name for display
        name_lower = model_name.lower()
        if "dino_vit" in name_lower:
            if "quantized" in name_lower:
                model_type = "Quantized DINO ViT"
            else:
                model_type = "DINO ViT"
        elif "earthgenome" in name_lower or "softcon" in name_lower:
            model_type = "EarthGenome"
        elif "google_satellite" in name_lower:
            model_type = "Google Satellite"
        else:
            # Fallback: use first part of model name
            model_type = model_name.split("_")[0].title()

        # Build final display name
        parts = [model_type + tile_info]
        if region:
            parts.append(region.replace("_", " ").title())

        return " - ".join(parts)

    def _infer_faiss_from_db(self, db_path: str) -> Optional[str]:
        candidate = pathlib.Path(db_path)
        base = candidate.stem
        name_candidates = {base}
        if base.endswith("_metadata"):
            name_candidates.add(base[: -len("_metadata")])

        patterns = []
        for name in name_candidates:
            patterns.extend(
                [
                    f"{name}.index",
                    f"{name}_faiss.index",
                    f"{name}_faiss*.index",
                    f"{name}*.index",
                ]
            )

        for pattern in patterns:
            matches = sorted(candidate.parent.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def _infer_geometry_from_db(self, db_path: str) -> Optional[str]:
        if not db_path:
            return None

        base_name = pathlib.Path(db_path).stem
        if base_name.endswith("_metadata"):
            base_name = base_name[: -len("_metadata")]

        candidates = [base_name]
        parts = base_name.split("_")
        if parts:
            candidates.append(parts[0])
            if len(parts) > 1:
                candidates.append("_".join(parts[:2]))

        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            geometry_path = self._resolve_geometry_path(candidate)
            if geometry_path:
                return geometry_path
        return None

    def _infer_geometry_cache_from_db(self, db_path: str) -> Optional[str]:
        """Infer the geometry cache Parquet URL from the database path.

        For S3/GCS URLs, constructs the cache URL by replacing _metadata.db or .db
        with _geometry_cache.parquet. For local paths, returns None (no cache needed).
        """
        if not db_path:
            return None

        # Only infer for remote databases
        if not self.is_remote_url(db_path):
            return None

        # Construct geometry cache URL from database URL
        # e.g., s3://bucket/path/name_metadata.db -> s3://bucket/path/name_geometry_cache.parquet
        if db_path.endswith("_metadata.db"):
            return db_path.replace("_metadata.db", "_geometry_cache.parquet")
        elif db_path.endswith(".db"):
            return db_path.replace(".db", "_geometry_cache.parquet")

        return None

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest_entries(self, manifest_path: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        try:
            with open(manifest_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    entries.append(
                        {
                            k: (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items()
                        }
                    )
        except Exception as exc:  # pragma: no cover
            if self.verbose:
                print(f"⚠️  Failed to read manifest at {manifest_path}: {exc}")
        return entries

    def _discover_available_models(
        self, directory_path: str, manifest_rows: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if not directory_path:
            return []

        dir_path = pathlib.Path(directory_path).expanduser()
        if not dir_path.exists():
            if self.verbose:
                print(f"⚠️  Database directory not found: {directory_path}")
            return []

        manifest_lookup = {
            (row.get("model_name") or "").strip(): row
            for row in manifest_rows
            if row.get("model_name")
        }

        discovered: List[Dict[str, str]] = []
        seen_db_paths = set()

        for model_name, row in manifest_lookup.items():
            artifacts = self._locate_model_artifacts(dir_path, model_name)
            if not artifacts:
                if self.verbose:
                    print(f"⚠️  Model files not found locally for {model_name}")
                continue

            if artifacts["db_path"] in seen_db_paths:
                continue

            region = (row.get("region") or "").strip() or None
            geometry_path = self._resolve_geometry_path(region)
            entry = {
                "model_name": model_name,
                "region": region,
                "db_path": artifacts["db_path"],
                "faiss_path": artifacts["faiss_path"],
                "display_name": self._format_model_display_name(region, model_name),
                "geometry_path": geometry_path,
            }
            discovered.append(entry)
            seen_db_paths.add(entry["db_path"])

        return discovered

    def _locate_model_artifacts(
        self, root_directory: pathlib.Path, model_name: str
    ) -> Optional[Dict[str, str]]:
        candidate_dirs = [root_directory / model_name, root_directory]
        for candidate in candidate_dirs:
            if not candidate.exists():
                continue

            db_path = self._match_single_file(
                candidate,
                [
                    (f"{model_name}_metadata.db", f"{model_name}_metadata.db"),
                    (f"{model_name}_metadata" + "*.db", f"{model_name}_metadata.db"),
                    (f"{model_name}.db", f"{model_name}.db"),
                ],
            )

            if not db_path:
                continue

            faiss_path = self._match_single_file(
                candidate,
                [
                    (f"{model_name}_faiss" + "*.index", None),
                    (f"{model_name}.index", f"{model_name}.index"),
                ],
            )

            if db_path and faiss_path:
                return {"db_path": db_path, "faiss_path": faiss_path}

        return None

    def _match_single_file(
        self, directory: pathlib.Path, pattern_specs: List[Tuple[str, Optional[str]]]
    ) -> Optional[str]:
        for pattern, preferred in pattern_specs:
            matches = sorted(directory.glob(pattern))
            selected = self._select_preferred_path(matches, preferred)
            if selected:
                return str(selected)
        return None

    @staticmethod
    def _select_preferred_path(
        matches: List[pathlib.Path], preferred_name: Optional[str] = None
    ) -> Optional[pathlib.Path]:
        if not matches:
            return None

        if preferred_name:
            for match in matches:
                if match.name == preferred_name:
                    return match

        matches = sorted(
            matches,
            key=lambda path: (
                DataManager._has_numeric_suffix(path.stem),
                len(path.name),
                path.name,
            ),
        )
        return matches[0]

    @staticmethod
    def _has_numeric_suffix(stem: str) -> int:
        return 1 if re.search(r"_\d+$", stem) else 0

    def _resolve_geometry_path(self, region: Optional[str]) -> Optional[str]:
        if not region or not self.geometries_dir:
            return None

        geom_dir = pathlib.Path(self.geometries_dir)
        if not geom_dir.exists():
            return None

        normalized = region.strip().lower().replace(" ", "_")
        variants = [
            normalized,
            normalized.replace("-", "_"),
            normalized.replace("_", "-"),
        ]

        for name in dict.fromkeys(variants):
            candidate = geom_dir / f"{name}.geojson"
            if candidate.exists():
                if self.verbose:
                    print(f"✅ Using geometry file: {candidate}")
                return str(candidate)

        normalized_full = normalized
        for geojson_path in geom_dir.glob("*.geojson"):
            stem = geojson_path.stem.lower()
            stem_variants = {stem, stem.replace("-", "_"), stem.replace("_", "-")}
            if any(
                normalized_full.startswith(variant) or variant in normalized_full
                for variant in stem_variants
            ):
                if self.verbose:
                    print(f"✅ Using geometry file: {geojson_path}")
                return str(geojson_path)

        if self.verbose:
            print(f"⚠️  No geometry found for region '{region}' in {geom_dir}")
        return None

    @staticmethod
    def _format_model_display_name(region: Optional[str], model_name: str) -> str:
        if region:
            return f"{region} / {model_name}"
        return model_name

    # ------------------------------------------------------------------
    # DuckDB helpers
    # ------------------------------------------------------------------

    def _connect_duckdb(self, database_path: str) -> duckdb.DuckDBPyConnection:
        if DatabaseConstants.is_gcs_path(database_path) and self.verbose:
            print(f"🌐 Connecting to GCS database: {database_path}")
            if os.getenv("GCS_ACCESS_KEY_ID"):
                print("🔑 Using HMAC key authentication")
            else:
                print("🔑 Using default Google Cloud authentication")
        elif DatabaseConstants.is_s3_path(database_path) and self.verbose:
            print(f"🌐 Connecting to S3 database: {database_path}")
            print("🔑 Using AWS credentials from environment/config")
        elif self.verbose:
            print(f"💾 Connecting to local database: {database_path}")

        try:
            connection = DatabaseConstants.setup_duckdb_connection(
                database_path, read_only=True
            )
            if self.verbose:
                print("✅ Database connection established successfully")
            return connection
        except Exception as exc:
            if DatabaseConstants.is_gcs_path(database_path):
                error_msg = f"Failed to connect to GCS database: {exc}"
                if (
                    "authentication" in str(exc).lower()
                    or "forbidden" in str(exc).lower()
                ):
                    error_msg += (
                        "\n💡 Check your GCS authentication setup (see GCS_SETUP.md)"
                    )
                raise RuntimeError(error_msg)
            elif DatabaseConstants.is_s3_path(database_path):
                error_msg = f"Failed to connect to S3 database: {exc}"
                if (
                    "authentication" in str(exc).lower()
                    or "forbidden" in str(exc).lower()
                    or "403" in str(exc)
                ):
                    error_msg += "\n💡 Check your AWS credentials (aws configure or environment variables)"
                raise RuntimeError(error_msg)
            raise RuntimeError(f"Failed to connect to local database: {exc}")

    def _apply_duckdb_settings(self, database_path: Optional[str]) -> None:
        self._apply_connection_runtime_settings(self.duckdb_connection)

        if database_path:
            extension_queries = DatabaseConstants.get_extension_setup_queries(
                database_path
            )
            for query in extension_queries:
                try:
                    self.duckdb_connection.execute(query)
                    if self.verbose:
                        if "httpfs" in query:
                            print("📦 httpfs extension loaded for GCS support")
                        elif "spatial" in query:
                            print("🗺️  spatial extension loaded for geometry support")
                except Exception as exc:
                    raise RuntimeError(f"Failed to load required extension: {exc}")

    def _apply_connection_runtime_settings(
        self, connection: Optional[duckdb.DuckDBPyConnection]
    ) -> None:
        if connection is None:
            return
        for query in DatabaseConstants.get_memory_setup_queries():
            connection.execute(query)
        try:
            connection.execute("SET enable_progress_bar=false")
            connection.execute("SET enable_profiling='no_output'")
            connection.execute("PRAGMA disable_profiling")
            connection.execute("SET enable_object_cache=false")
            if self.verbose:
                print("✅ Progress bar and profiling disabled")
        except Exception:  # pragma: no cover - optional settings
            pass

    def _refresh_id_columns(self) -> None:
        columns = self._detect_id_columns()
        self.id_column_candidates = columns
        for candidate in ("source_id", "tile_id"):
            if candidate in columns:
                self.external_id_column = candidate
                return
        self.external_id_column = "id"

    def _detect_id_columns(self) -> List[str]:
        try:
            rows = self.duckdb_connection.execute(
                "PRAGMA table_info('geo_embeddings')"
            ).fetchall()
        except Exception:
            return ["id"]
        columns = [row[1] for row in rows if len(row) > 1]
        candidates = [col for col in ("source_id", "tile_id", "id") if col in columns]
        if candidates:
            return candidates
        return ["id"]

    def _detect_embedding_dim(self) -> int:
        try:
            embedding_dim = DatabaseConstants.detect_embedding_dimension(
                self.duckdb_connection
            )
            if self.verbose:
                print(f"🔍 Detected embedding dimension: {embedding_dim}")
            return embedding_dim
        except ValueError as exc:
            if self.verbose:
                print(f"⚠️ Could not detect embedding dimension: {exc}")
                print("⚠️ Using default dimension of 384")
            return 384

    def _detect_embedding_type(self) -> str:
        try:
            row = self.duckdb_connection.execute(
                "SELECT typeof(embedding) FROM geo_embeddings LIMIT 1"
            ).fetchone()
            if row and row[0]:
                emb_type = str(row[0])
                if self.verbose:
                    print(f"🔍 Detected embedding type: {emb_type}")
                return emb_type
        except Exception:
            pass
        return "FLOAT[]"

    def _resolve_embedding_select_expr(self) -> str:
        env_expr = os.getenv("GEOVIBES_EMBEDDING_SELECT_EXPR")
        if env_expr and env_expr.strip():
            return env_expr.strip()

        emb_type = (self.embedding_type or "").upper()
        # Fixed-size arrays/lists can be converted to float32 in Python without SQL casts.
        if "[" in emb_type:
            return "embedding"
        return "CAST(embedding AS FLOAT[])"

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _mem_available_mb() -> Optional[float]:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split(":", 1)[1].strip().split()[0])
                        return kb / 1024.0
        except Exception:
            return None
        return None

    def suggest_background_pool_size(self, requested: Optional[int] = None) -> int:
        """Suggest a memory-safe background pool size for remote prefetch."""
        target_default = self._env_int("GEOVIBES_PREFETCH_WORKER_TARGET", 44)
        cap_default = self._env_int("GEOVIBES_PREFETCH_WORKER_CAP", 44)
        floor_default = self._env_int("GEOVIBES_PREFETCH_WORKER_FLOOR", 16)
        mem_per_conn_mb = max(
            16, self._env_int("GEOVIBES_PREFETCH_WORKER_MEM_MB", 96)
        )

        target = max(1, int(requested if requested is not None else target_default))
        cap = max(target, int(cap_default))
        floor = max(1, int(floor_default))
        size = min(target, cap)

        mem_available_mb = self._mem_available_mb()
        mem_cap: Optional[int] = None
        if mem_available_mb is not None and mem_per_conn_mb > 0:
            mem_cap = max(1, int(mem_available_mb // mem_per_conn_mb))
            size = min(size, mem_cap)

        if mem_cap is not None:
            floor = min(floor, mem_cap)
        floor = min(floor, cap)
        size = max(1, min(cap, max(size, floor)))
        return int(size)

    def _warm_up_remote_database(self) -> None:
        """Warm up remote database by preloading row groups.

        For httpfs databases, the first query to each row group is slow (~500ms).
        This method preloads data by:
        1. Running a spatial nearest-point query (used when labeling points)
        2. Fetching an embedding (used when updating the query vector)

        This ensures the first user interaction is fast.
        """
        import time

        try:
            print("🔄 Warming up remote database (this may take a moment)...")
            start = time.perf_counter()

            # Get database centroid for warmup queries
            centroid = self.duckdb_connection.execute(
                "SELECT AVG(ST_X(geometry)), AVG(ST_Y(geometry)) FROM geo_embeddings LIMIT 10000"
            ).fetchone()
            center_lon, center_lat = centroid if centroid else (0, 0)

            # Warm up the nearest_point query (spatial scan + embedding fetch)
            # This is the query that runs when user clicks to label a point
            print("   Loading spatial index...", end="", flush=True)
            spatial_start = time.perf_counter()
            self.duckdb_connection.execute(
                DatabaseConstants.NEAREST_POINT_QUERY, [center_lon, center_lat]
            ).fetchone()
            spatial_time = time.perf_counter() - spatial_start
            print(f" done ({spatial_time:.1f}s)")

            elapsed = time.perf_counter() - start
            print(f"✅ Database ready ({elapsed:.1f}s)")

        except Exception as exc:
            if self.verbose:
                print(f"⚠️  Database warm-up failed: {exc}")

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def _setup_boundary_and_center(self):
        boundary_path = self.current_geometry_path
        if (
            not boundary_path
            and self.current_database_info
            and self.current_database_info.get("geometry_path")
        ):
            boundary_path = self.current_database_info.get("geometry_path")

        # For remote databases, fetch boundary from S3 if available
        if (
            not boundary_path
            and self.current_database_info
            and self.current_database_info.get("boundary_url")
        ):
            boundary_path = self._fetch_remote_boundary(
                self.current_database_info["boundary_url"]
            )

        if boundary_path:
            try:
                boundary_gdf = gpd.read_file(boundary_path)
                center_y, center_x = (
                    boundary_gdf.geometry.iloc[0].centroid.y,
                    boundary_gdf.geometry.iloc[0].centroid.x,
                )
                if self.verbose:
                    print(f"📍 Using boundary file: {boundary_path}")
                return boundary_path, (center_y, center_x)
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Could not load boundary file {boundary_path}: {exc}")
                    print("⚠️  Using database centroid for centering")

        center_y, center_x = get_database_centroid(
            self.duckdb_connection, verbose=self.verbose
        )
        return None, (center_y, center_x)

    def _fetch_remote_boundary(self, boundary_url: str) -> Optional[str]:
        """Fetch and cache a remote boundary GeoJSON file."""
        import hashlib

        import fsspec

        # Create cache directory
        cache_dir = pathlib.Path.home() / ".cache" / "geovibes" / "boundaries"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename from URL hash
        url_hash = hashlib.md5(boundary_url.encode()).hexdigest()[:16]
        # Extract filename from URL for readability
        url_filename = boundary_url.split("/")[-1]
        cache_path = cache_dir / f"{url_hash}_{url_filename}"

        if cache_path.exists():
            if self.verbose:
                print(f"📍 Using cached boundary: {cache_path}")
            return str(cache_path)

        if self.verbose:
            print(f"📥 Downloading boundary from {boundary_url}...")

        try:
            # Use fsspec to handle S3 URLs
            with fsspec.open(boundary_url, "rb", anon=True) as f:
                content = f.read()
            cache_path.write_bytes(content)
            if self.verbose:
                print(f"📍 Cached boundary to: {cache_path}")
            return str(cache_path)
        except Exception as exc:
            if self.verbose:
                print(f"⚠️  Could not fetch boundary from {boundary_url}: {exc}")
            return None

    def _load_faiss_index(self, faiss_path: str) -> faiss.Index:
        """Load FAISS index from local path or S3 URL.

        For S3 URLs, uses FaissCache to download and cache the index locally.

        Args:
            faiss_path: Local path or S3 URL to FAISS index

        Returns:
            Loaded FAISS index
        """
        if DatabaseConstants.is_s3_path(faiss_path):
            if self.verbose:
                print("📥 Downloading FAISS index from S3 (with caching)...")
            cache = FaissCache()
            return cache.get_index(faiss_path, show_progress=True)
        else:
            return faiss.read_index(faiss_path)

    def _load_geometry_cache(self, geometry_cache_url: str) -> None:
        """Download and connect to geometry cache for fast metadata queries.

        For remote databases, the geometry cache is a small Parquet file containing
        just id + geometry columns. This enables fast local queries instead of
        fetching scattered IDs over httpfs.

        Args:
            geometry_cache_url: S3 or GCS URL to the geometry cache Parquet file
        """
        if self.verbose:
            print("📥 Downloading geometry cache (with caching)...")

        cache = FaissCache()
        local_path = cache.get_geometry_cache(geometry_cache_url, show_progress=True)
        self._geometry_cache_local_path = local_path

        if self.verbose:
            print(f"🗺️  Connecting to geometry cache: {local_path}")

        # Create a separate DuckDB connection for geometry queries
        self._geometry_cache_connection = duckdb.connect(":memory:")
        self._geometry_cache_connection.execute("INSTALL spatial; LOAD spatial;")

        # Create a view to query the Parquet file
        self._geometry_cache_connection.execute(
            f"CREATE VIEW geometry_cache AS SELECT * FROM '{local_path}'"
        )

        row_count = self._geometry_cache_connection.execute(
            "SELECT COUNT(*) FROM geometry_cache"
        ).fetchone()[0]

        if self.verbose:
            print(f"✅ Geometry cache ready ({row_count:,} geometries)")

    def _close_auxiliary_connections(self) -> None:
        self._close_background_connection_pool()

        if getattr(self, "_geometry_cache_connection", None):
            try:
                self._geometry_cache_connection.close()
            except Exception:
                pass
        self._geometry_cache_connection = None
        self._geometry_cache_local_path = None

        if getattr(self, "_row_group_cache_connection", None):
            try:
                self._row_group_cache_connection.close()
            except Exception:
                pass
        self._row_group_cache_connection = None
        self._row_group_cache_local_path = None

    def _row_group_cache_path(
        self, database_path: str, row_group_size: int
    ) -> pathlib.Path:
        cache_dir = pathlib.Path.home() / ".cache" / "geovibes" / "row_groups"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(
            f"{database_path}|{int(row_group_size)}".encode("utf-8")
        ).hexdigest()[:16]
        return cache_dir / f"{cache_key}.parquet"

    def _load_row_group_cache(self, row_group_size: int = 122880) -> None:
        """Create/load local id->row_group cache for remote prefetch batching."""
        if not self.current_database_path or not self.is_remote_url(
            self.current_database_path
        ):
            return
        if self.duckdb_connection is None:
            return

        cache_path = self._row_group_cache_path(self.current_database_path, row_group_size)
        if not cache_path.exists():
            if self.verbose:
                print("📦 Building row-group cache for remote prefetch...")
            self.duckdb_connection.execute(
                f"""
                COPY (
                    SELECT id, CAST(FLOOR(rowid / {int(row_group_size)}) AS BIGINT) AS row_group
                    FROM remote_db.geo_embeddings
                )
                TO '{cache_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )

        self._row_group_cache_connection = duckdb.connect(":memory:")
        self._row_group_cache_connection.execute(
            f"CREATE VIEW row_group_cache AS SELECT * FROM '{cache_path}'"
        )
        self._row_group_cache_local_path = cache_path
        self._row_group_size = int(row_group_size)

        if self.verbose:
            rows = self._row_group_cache_connection.execute(
                "SELECT COUNT(*) FROM row_group_cache"
            ).fetchone()[0]
            print(f"✅ Row-group cache ready ({rows:,} ids)")

    def _close_background_connection_pool(self) -> None:
        self._ensure_background_pool_state()

        with self._background_pool_lock:
            pool = self._background_pool_queue
            self._background_pool_queue = None
            self._background_pool_size = 0
            self._background_pool_created = 0

        if pool is None:
            return

        while True:
            try:
                conn = pool.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass

    def _ensure_background_pool_state(self) -> None:
        if not hasattr(self, "_background_pool_lock"):
            self._background_pool_lock = threading.Lock()
        if not hasattr(self, "_background_pool_queue"):
            self._background_pool_queue = None
        if not hasattr(self, "_background_pool_size"):
            self._background_pool_size = 0
        if not hasattr(self, "_background_pool_created"):
            self._background_pool_created = 0

    def configure_background_connection_pool(self, max_connections: int) -> None:
        """Configure persistent pool for remote background embedding fetches."""
        self._ensure_background_pool_state()

        target_size = max(0, int(max_connections))

        if (
            target_size == 0
            or not self.current_database_path
            or not self.is_remote_url(self.current_database_path)
        ):
            self._close_background_connection_pool()
            return

        with self._background_pool_lock:
            if (
                self._background_pool_queue is not None
                and self._background_pool_size == target_size
            ):
                return

        self._close_background_connection_pool()
        with self._background_pool_lock:
            self._background_pool_queue = queue.LifoQueue(maxsize=target_size)
            self._background_pool_size = target_size
            self._background_pool_created = 0

    def precreate_background_connection_pool(
        self, max_connections: int, n_workers: Optional[int] = None
    ) -> Dict[str, float]:
        """Eagerly pre-create background connections to reduce first-search latency."""
        self.configure_background_connection_pool(max_connections)
        stats_before = self.background_pool_stats()
        target_size = int(stats_before.get("size", 0))
        if target_size <= 0:
            return {
                "target_size": 0.0,
                "created_ok": 0.0,
                "elapsed_ms": 0.0,
                "idle": 0.0,
            }

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        workers = max(1, min(int(n_workers or target_size), target_size))
        start = time.perf_counter()

        def _prime_once() -> int:
            conn = self.acquire_background_connection(timeout_seconds=60.0)
            if conn is None:
                return 0
            self.release_background_connection(conn)
            return 1

        created_ok = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_prime_once) for _ in range(target_size)]
            for future in as_completed(futures):
                try:
                    created_ok += int(future.result())
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        stats_after = self.background_pool_stats()
        if self.verbose:
            print(
                "✅ Background pool pre-created "
                f"(size={stats_after['size']}, created={stats_after['created']}, idle={stats_after['idle']}, "
                f"elapsed={elapsed_ms:.1f}ms)"
            )

        return {
            "target_size": float(target_size),
            "created_ok": float(created_ok),
            "elapsed_ms": float(elapsed_ms),
            "idle": float(stats_after.get("idle", 0)),
        }

    def start_background_pool_precreation(
        self, max_connections: int, n_workers: Optional[int] = None
    ) -> Optional[threading.Thread]:
        """Start background pool pre-creation asynchronously."""
        if (
            max_connections <= 0
            or not self.current_database_path
            or not self.is_remote_url(self.current_database_path)
        ):
            return None

        thread = threading.Thread(
            target=self.precreate_background_connection_pool,
            kwargs={
                "max_connections": int(max_connections),
                "n_workers": int(n_workers) if n_workers is not None else None,
            },
            daemon=True,
        )
        thread.start()
        return thread

    def acquire_background_connection(
        self, timeout_seconds: float = 30.0
    ) -> Optional[duckdb.DuckDBPyConnection]:
        """Acquire pooled background connection (or create one if pool disabled)."""
        self._ensure_background_pool_state()

        with self._background_pool_lock:
            pool = self._background_pool_queue
            pool_size = self._background_pool_size
            if pool is None or pool_size <= 0:
                return self.create_background_connection()

            if self._background_pool_created < pool_size:
                self._background_pool_created += 1
                should_create = True
            else:
                should_create = False

        if should_create:
            conn = self.create_background_connection()
            if conn is None:
                with self._background_pool_lock:
                    self._background_pool_created = max(
                        0, self._background_pool_created - 1
                    )
            return conn

        try:
            return pool.get(timeout=max(0.01, float(timeout_seconds)))
        except queue.Empty:
            if self.verbose:
                print("⚠️ Background pool timeout; creating transient connection")
            return self.create_background_connection()

    def release_background_connection(
        self, connection: Optional[duckdb.DuckDBPyConnection]
    ) -> None:
        """Return a background connection to pool (or close if not pooled)."""
        self._ensure_background_pool_state()

        if connection is None:
            return

        with self._background_pool_lock:
            pool = self._background_pool_queue
            pool_size = self._background_pool_size

        if pool is None or pool_size <= 0:
            try:
                connection.close()
            except Exception:
                pass
            return

        try:
            pool.put_nowait(connection)
        except queue.Full:
            # Connection is transient or pool was resized.
            try:
                connection.close()
            except Exception:
                pass

    def background_pool_stats(self) -> Dict[str, int]:
        self._ensure_background_pool_state()

        with self._background_pool_lock:
            size = self._background_pool_size
            created = self._background_pool_created
            idle = self._background_pool_queue.qsize() if self._background_pool_queue else 0
        return {"size": size, "created": created, "idle": idle}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._close_auxiliary_connections()
        if getattr(self, "_owns_connection", False):
            if getattr(self, "duckdb_connection", None):
                self.duckdb_connection.close()
                if self.verbose:
                    print("🔌 DuckDB connection closed.")

    def create_background_connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """Create a separate connection for background operations."""
        if not self.current_database_path:
            return None
        connection = self._connect_duckdb(self.current_database_path)
        self._apply_connection_runtime_settings(connection)
        return connection

    def _build_id_predicate(
        self,
        ids: List[object],
        *,
        query_mode: str,
        target_alias: str = "",
        id_column: str = "id",
    ) -> tuple[str, List[object]]:
        """Build an ID predicate SQL fragment and parameters."""
        normalized_mode = self._normalize_query_mode(query_mode)
        prefix = f"{target_alias}." if target_alias else ""
        if normalized_mode == "values_join":
            values_rows = ",".join(["(?)" for _ in ids])
            clause = (
                f"{prefix}{id_column} IN ("
                f"SELECT CAST(v.id AS BIGINT) FROM (VALUES {values_rows}) AS v(id)"
                ")"
            )
            return clause, ids

        placeholders = ",".join(["?" for _ in ids])
        clause = f"{prefix}{id_column} IN ({placeholders})"
        return clause, ids

    def group_ids_for_prefetch(
        self, ids: List[int], row_group_size: Optional[int] = None
    ) -> Tuple[List[List[int]], str, float]:
        """Group IDs for remote embedding prefetch.

        Uses local row-group cache when available (robust to table reordering),
        otherwise falls back to id//row_group_size grouping.
        """
        if not ids:
            return [], "none", 0.0

        int_ids = [int(id_) for id_ in ids]
        effective_row_group = int(row_group_size or self._row_group_size)

        if self._row_group_cache_connection is not None:
            import time

            placeholders = ",".join(["?" for _ in int_ids])
            start = time.perf_counter()
            df = self._row_group_cache_connection.execute(
                f"SELECT id, row_group FROM row_group_cache WHERE id IN ({placeholders})",
                int_ids,
            ).fetchdf()
            lookup_ms = (time.perf_counter() - start) * 1000.0

            groups: dict[int, list[int]] = defaultdict(list)
            found = set()
            for _, row in df.iterrows():
                id_val = int(row["id"])
                groups[int(row["row_group"])].append(id_val)
                found.add(id_val)

            # Defensive fallback if cache misses IDs.
            for id_val in int_ids:
                if id_val not in found:
                    groups[id_val // effective_row_group].append(id_val)

            batches = [sorted(batch) for batch in groups.values() if batch]
            scheduler = self._normalize_batch_scheduler(
                getattr(self, "_prefetch_batch_scheduler", "id_ascending")
            )
            if scheduler == "id_ascending":
                batches.sort(key=lambda batch: batch[0])
            elif scheduler == "id_descending":
                batches.sort(key=lambda batch: batch[0], reverse=True)
            return batches, "row_group_cache", lookup_ms

        groups = defaultdict(list)
        for id_val in int_ids:
            groups[id_val // effective_row_group].append(id_val)
        batches = [sorted(batch) for batch in groups.values() if batch]
        scheduler = self._normalize_batch_scheduler(
            getattr(self, "_prefetch_batch_scheduler", "id_ascending")
        )
        if scheduler == "id_ascending":
            batches.sort(key=lambda batch: batch[0])
        elif scheduler == "id_descending":
            batches.sort(key=lambda batch: batch[0], reverse=True)
        return batches, "id_div", 0.0

    def fetch_embeddings(
        self,
        point_ids: List[str],
        chunk_size: Optional[int] = None,
        include_geometry: bool = True,
    ):
        if not point_ids:
            return

        chunk_size = chunk_size or DatabaseConstants.EMBEDDING_CHUNK_SIZE
        if len(point_ids) > 100 and self.verbose:
            print(f"🔄 Fetching embeddings for {len(point_ids)} points...")

        for i in range(0, len(point_ids), chunk_size):
            chunk = point_ids[i : i + chunk_size]
            prepared_chunk = prepare_ids_for_query(chunk)
            placeholders = ",".join(["?" for _ in prepared_chunk])
            select_parts = ["id"]
            external_column = getattr(self, "external_id_column", "id")
            if external_column != "id":
                select_parts.append(external_column)
            select_expr = getattr(self, "_embedding_select_expr", "embedding")
            select_parts.append(f"{select_expr} as embedding")
            if include_geometry:
                select_parts.append("geometry")
            select_clause = ", ".join(select_parts)
            query = f"""
            SELECT {select_clause}
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
            log_to_file(
                f"Fetch embeddings: Built query for chunk with IDs: {prepared_chunk}"
            )
            arrow_table = self.duckdb_connection.execute(
                query, prepared_chunk
            ).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()
            yield chunk_df

    def fetch_embeddings_with_connection(
        self,
        connection: duckdb.DuckDBPyConnection,
        point_ids: List[str],
        chunk_size: Optional[int] = None,
        include_geometry: bool = True,
    ):
        """Fetch embeddings using a specific connection (for background operations)."""
        if not point_ids:
            return

        chunk_size = chunk_size or DatabaseConstants.EMBEDDING_CHUNK_SIZE

        for i in range(0, len(point_ids), chunk_size):
            chunk = point_ids[i : i + chunk_size]
            prepared_chunk = prepare_ids_for_query(chunk)
            placeholders = ",".join(["?" for _ in prepared_chunk])
            select_parts = ["id"]
            external_column = getattr(self, "external_id_column", "id")
            if external_column != "id":
                select_parts.append(external_column)
            select_expr = getattr(self, "_embedding_select_expr", "embedding")
            select_parts.append(f"{select_expr} as embedding")
            if include_geometry:
                select_parts.append("geometry")
            select_clause = ", ".join(select_parts)
            query = f"""
            SELECT {select_clause}
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
            arrow_table = connection.execute(query, prepared_chunk).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()
            yield chunk_df

    def fetch_embedding_map(
        self, point_ids: List[str], chunk_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Fetch embeddings into an id->np.ndarray map using the primary connection."""
        if self.duckdb_connection is None:
            return {}
        return self.fetch_embedding_map_with_connection(
            self.duckdb_connection,
            point_ids,
            chunk_size=chunk_size,
        )

    def fetch_embedding_map_with_connection(
        self,
        connection: duckdb.DuckDBPyConnection,
        point_ids: List[str],
        chunk_size: Optional[int] = None,
        query_mode: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Fetch embeddings into an id->np.ndarray map using the provided connection."""
        if not point_ids:
            return {}

        chunk_size = chunk_size or DatabaseConstants.EMBEDDING_CHUNK_SIZE
        out: Dict[str, np.ndarray] = {}
        effective_mode = self._normalize_query_mode(
            query_mode or self._prefetch_query_mode
        )

        for i in range(0, len(point_ids), chunk_size):
            chunk = point_ids[i : i + chunk_size]
            prepared_chunk = self._coerce_id_list(prepare_ids_for_query(chunk))
            predicate, params = self._build_id_predicate(
                prepared_chunk,
                query_mode=effective_mode,
                id_column="id",
            )
            select_expr = getattr(self, "_embedding_select_expr", "embedding")
            query = (
                f"SELECT id, {select_expr} as embedding "
                "FROM geo_embeddings "
                f"WHERE {predicate}"
            )
            rows = connection.execute(query, params).fetchall()
            for id_val, embedding in rows:
                out[str(id_val)] = np.asarray(embedding, dtype=np.float32)

        return out

    def nearest_point(self, lon: float, lat: float):
        import time

        log_to_file(f"nearest_point: START (lon={lon:.4f}, lat={lat:.4f})")
        sql = DatabaseConstants.NEAREST_POINT_QUERY
        params = [lon, lat]

        exec_start = time.perf_counter()
        cursor = self.duckdb_connection.execute(sql, params)
        exec_time = (time.perf_counter() - exec_start) * 1000
        log_to_file(f"nearest_point: execute() completed in {exec_time:.1f}ms")

        fetch_start = time.perf_counter()
        result = cursor.fetchone()
        fetch_time = (time.perf_counter() - fetch_start) * 1000
        log_to_file(f"nearest_point: fetchone() completed in {fetch_time:.1f}ms")

        total_time = exec_time + fetch_time
        log_to_file(f"nearest_point: DONE total={total_time:.1f}ms")
        return result

    def query_geometries(self, ids: List[str]):
        if not ids:
            return None
        prepared_ids = prepare_ids_for_query(ids)
        placeholders = ",".join(["?" for _ in prepared_ids])
        sql = f"""
        SELECT ST_AsGeoJSON(geometry) as geometry
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        return self.duckdb_connection.execute(sql, prepared_ids).df()

    def query_search_metadata(
        self, faiss_ids: List[int], query_mode: Optional[str] = None
    ):
        if not faiss_ids:
            return None
        prepared_ids = self._coerce_id_list([int(id_) for id_ in faiss_ids])
        effective_mode = self._normalize_query_mode(
            query_mode or self._metadata_query_mode
        )

        # Use local geometry cache for remote databases (much faster)
        if self._geometry_cache_connection is not None:
            predicate, params = self._build_id_predicate(
                prepared_ids,
                query_mode=effective_mode,
                id_column="id",
            )
            sql = (
                "SELECT id, "
                "ST_AsGeoJSON(geometry) AS geometry_json, "
                "ST_AsText(geometry) AS geometry_wkt "
                "FROM geometry_cache "
                f"WHERE {predicate}"
            )
            return self._geometry_cache_connection.execute(sql, params).fetchdf()

        # Fall back to remote database query
        select_parts = ["id"]
        external_column = getattr(self, "external_id_column", "id")
        if external_column != "id":
            select_parts.append(external_column)
        select_parts.extend(
            [
                "ST_AsGeoJSON(geometry) AS geometry_json",
                "ST_AsText(geometry) AS geometry_wkt",
            ]
        )
        select_clause = ", ".join(select_parts)
        predicate, params = self._build_id_predicate(
            prepared_ids,
            query_mode=effective_mode,
            id_column="id",
        )
        sql = (
            f"SELECT {select_clause} "
            "FROM geo_embeddings "
            f"WHERE {predicate}"
        )
        return self.duckdb_connection.execute(sql, params).fetchdf()

    def switch_database(self, database_path: str):
        if database_path == self.current_database_path:
            return

        self._close_auxiliary_connections()

        self.current_database_path = database_path
        self.current_database_info = self.database_info_by_path.get(database_path)
        if self.current_database_info:
            self.current_faiss_path = self.current_database_info["faiss_path"]
            self.current_geometry_path = self.current_database_info.get("geometry_path")
            self.current_geometry_cache_path = self.current_database_info.get(
                "geometry_cache_path"
            )
            self.tile_spec = self.current_database_info.get("tile_spec")
        else:
            self.current_faiss_path = None
            self.current_geometry_path = None
            self.current_geometry_cache_path = None
            self.tile_spec = None

        if not self.tile_spec:
            self.tile_spec = infer_tile_spec_from_name(database_path)

        if getattr(self, "_owns_connection", False):
            if getattr(self, "duckdb_connection", None):
                self.duckdb_connection.close()

        self.duckdb_connection = self._connect_duckdb(database_path)
        self._owns_connection = True
        self._apply_duckdb_settings(database_path)
        self._refresh_id_columns()

        if self.current_geometry_cache_path and self.is_remote_url(database_path):
            self._load_geometry_cache(self.current_geometry_cache_path)
        if self.is_remote_url(database_path):
            self._load_row_group_cache(self._row_group_size)

        if not self.current_faiss_path:
            raise RuntimeError(
                f"No FAISS index recorded for {os.path.basename(database_path)}"
            )

        self.faiss_index = self._load_faiss_index(self.current_faiss_path)
        self.embedding_dim = self._detect_embedding_dim()
        self.embedding_type = self._detect_embedding_type()
        self._embedding_select_expr = self._resolve_embedding_select_expr()
        pool_precreate_thread: Optional[threading.Thread] = None
        if DatabaseConstants.is_gcs_path(database_path) or DatabaseConstants.is_s3_path(
            database_path
        ):
            pool_target = self.suggest_background_pool_size()
            pool_precreate_thread = self.start_background_pool_precreation(
                pool_target,
                n_workers=pool_target,
            )
            self._warm_up_remote_database()
            if pool_precreate_thread is not None and pool_precreate_thread.is_alive():
                pool_precreate_thread.join(timeout=10.0)

        self.effective_boundary_path, (self.center_y, self.center_x) = (
            self._setup_boundary_and_center()
        )


__all__ = ["DataManager"]
