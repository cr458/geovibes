"""Core business logic for the GeoVibes application."""

import json
import os
import warnings
from datetime import datetime
from typing import Dict, Optional

import duckdb
import ee
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point

from geovibes.ee_tools import (
    get_s2_rgb_median,
    get_s2_ndvi_median,
    get_s2_ndwi_median,
    get_ee_image_url,
    initialize_ee_with_credentials,
)
from geovibes.ui_config import (
    UIConstants,
    BasemapConfig,
    GeoVibesConfig,
    DatabaseConstants,
)
from geovibes.utils import list_databases_in_directory, get_database_centroid

warnings.simplefilter("ignore", category=FutureWarning)

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn(
        "MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key."
    )


class GeoVibesLogic:
    """Backend logic for GeoVibes, independent of UI framework."""

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        gcp_project: Optional[str] = None,
        duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GeoVibes logic.

        Args:
            duckdb_path: Path to DuckDB database file.
            duckdb_directory: Directory containing multiple DuckDB database files.
            boundary_path: Path to boundary GeoJSON file.
            start_date: Start date in YYYY-MM-DD format for Earth Engine basemaps.
            end_date: End date in YYYY-MM-DD format for Earth Engine basemaps.
            gcp_project: Google Cloud Project ID for Earth Engine authentication.
            duckdb_connection: Existing DuckDB connection to reuse.
            config: Configuration dictionary (deprecated, use individual parameters).
            config_path: Path to JSON configuration file (deprecated, use individual parameters).
            verbose: Enable detailed progress messages.
            **kwargs: Additional arguments for backwards compatibility.

        Raises:
            ValueError: If no database is available given the provided parameters.
            FileNotFoundError: If no .db files are found in the provided directory.
            RuntimeError: If there is an error connecting to the database.
        """
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes logic...")

        # Handle backwards compatibility with config files
        if config_path is not None:
            if self.verbose:
                print(
                    "⚠️  config_path is deprecated. Use individual parameters instead."
                )
            self.config = GeoVibesConfig.from_file(config_path)
            self.config.validate()
        elif config is not None:
            if self.verbose:
                print(
                    "⚠️  config dict is deprecated. Use individual parameters instead."
                )
            self.config = GeoVibesConfig.from_dict(config)
            self.config.validate()
        else:
            # Only validate if we have the minimum required parameters
            if (
                duckdb_path is None
                and duckdb_directory is None
                and duckdb_connection is None
            ):
                raise ValueError(
                    "Either duckdb_path, duckdb_directory, or duckdb_connection must be provided"
                )

            # Use individual parameters to create config
            self.config = GeoVibesConfig(
                duckdb_path=duckdb_path,
                duckdb_directory=duckdb_directory,
                boundary_path=boundary_path,
                start_date=start_date or "2024-01-01",
                end_date=end_date or "2025-01-01",
                gcp_project=gcp_project,
            )

            self.config.validate()

        self.ee_available = initialize_ee_with_credentials(self.config.gcp_project)

        # Initialize database list if directory is provided
        self.available_databases = []
        self.current_database_path = None
        if self.config.duckdb_directory:
            self.available_databases = list_databases_in_directory(
                self.config.duckdb_directory, verbose=self.verbose
            )
            if self.available_databases:
                self.current_database_path = self.available_databases[0]
                if self.verbose:
                    print(
                        f"📁 Found {len(self.available_databases)} databases in directory"
                    )
            else:
                raise FileNotFoundError("⚠️  No .db files found in directory")
        elif self.config.duckdb_path:
            self.current_database_path = self.config.duckdb_path

        if duckdb_connection is None:
            if self.current_database_path is None:
                raise ValueError("No database available given the provided parameters")

            # Show connection status for GCS paths
            if DatabaseConstants.is_gcs_path(self.current_database_path):
                if self.verbose:
                    print(
                        f"🌐 Connecting to GCS database: {self.current_database_path}"
                    )
                    import os

                    if os.getenv("GCS_ACCESS_KEY_ID"):
                        print("🔑 Using HMAC key authentication")
                    else:
                        print("🔑 Using default Google Cloud authentication")
            elif self.verbose:
                print(f"💾 Connecting to local database: {self.current_database_path}")

            try:
                self.duckdb_connection = DatabaseConstants.setup_duckdb_connection(
                    self.current_database_path, read_only=True
                )
                self._owns_connection = True

                if self.verbose:
                    print("✅ Database connection established successfully")
            except Exception as e:
                if DatabaseConstants.is_gcs_path(self.current_database_path):
                    error_msg = f"Failed to connect to GCS database: {str(e)}"
                    if (
                        "authentication" in str(e).lower()
                        or "forbidden" in str(e).lower()
                    ):
                        error_msg += "\n💡 Check your GCS authentication setup (see GCS_SETUP.md)"
                    raise RuntimeError(error_msg)
                else:
                    raise RuntimeError(f"Failed to connect to local database: {str(e)}")

            # Configure memory limits to prevent kernel crashes
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False

        if self.ee_available:
            try:
                self.ee_boundary = ee.Geometry(
                    shapely.geometry.mapping(
                        gpd.read_file(self.config.boundary_path).union_all()
                    )
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to create Earth Engine boundary: {e}")
                    print("⚠️  NDVI/NDWI basemaps will be unavailable")
                self.ee_boundary = None
        else:
            self.ee_boundary = None

        # Setup extensions in DuckDB (spatial and httpfs if needed)
        if self.current_database_path:
            extension_queries = DatabaseConstants.get_extension_setup_queries(
                self.current_database_path
            )
            for query in extension_queries:
                try:
                    self.duckdb_connection.execute(query)
                    if self.verbose and "httpfs" in query:
                        print("📦 httpfs extension loaded for GCS support")
                    elif self.verbose and "spatial" in query:
                        print("🗺️  spatial extension loaded for geometry support")
                except Exception as e:
                    if "httpfs" in query:
                        raise RuntimeError(
                            f"Failed to load httpfs extension for GCS support: {str(e)}"
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to load required extension: {str(e)}"
                        )

        # Detect embedding dimension from database
        try:
            self.embedding_dim = DatabaseConstants.detect_embedding_dimension(
                self.duckdb_connection
            )
            if self.verbose:
                print(f"🔍 Detected embedding dimension: {self.embedding_dim}")
        except ValueError as e:
            if self.verbose:
                print(f"⚠️ Could not detect embedding dimension: {e}")
                print("⚠️ Using default dimension of 1000")
            self.embedding_dim = 384

        # Warm up GCS database with initial search for better performance
        if DatabaseConstants.is_gcs_path(self.current_database_path):
            self._warm_up_gcs_database()

        # Get map center and set up boundary path
        self.center_y, self.center_x = self._setup_boundary_and_center()

        # Add Earth Engine basemap options (if available)
        self.basemap_tiles = {}
        self._setup_ee_basemaps()

        # Initialize state
        self.current_label = "Positive"
        self.select_val = UIConstants.POSITIVE_LABEL  # Initialize to positive
        self.pos_ids = []
        self.neg_ids = []
        self.detection_gdf = None
        self.lasso_mode = False
        self.query_vector = None
        self.detection_ids = []
        self.cached_embeddings = {}
        self.detections_with_embeddings = None
        self.vector_layer_data = None

    def _setup_ee_basemaps(self) -> None:
        """Set up Earth Engine basemaps (Sentinel-2 RGB, NDVI, NDWI) if available."""
        self.basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()

        if self.ee_available and self.ee_boundary is not None:
            try:
                if self.verbose:
                    print("🛰️ Setting up Earth Engine basemaps (S2 RGB, NDVI, NDWI)...")

                s2_rgb_median = get_s2_rgb_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date
                )
                s2_rgb_url = get_ee_image_url(
                    s2_rgb_median, BasemapConfig.S2_RGB_VIS_PARAMS
                )
                self.basemap_tiles["S2_RGB"] = s2_rgb_url

                ndvi_median = get_s2_ndvi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date
                )
                ndvi_url = get_ee_image_url(ndvi_median, BasemapConfig.NDVI_VIS_PARAMS)
                self.basemap_tiles["NDVI"] = ndvi_url

                ndwi_median = get_s2_ndwi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date
                )
                ndwi_url = get_ee_image_url(ndwi_median, BasemapConfig.NDWI_VIS_PARAMS)
                self.basemap_tiles["NDWI"] = ndwi_url

                if self.verbose:
                    print("✅ Earth Engine basemaps added successfully!")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to create Earth Engine basemaps: {e}")
                    print("⚠️  Continuing with basic basemaps only")
        else:
            if not self.ee_available and self.verbose:
                print("⚠️  Earth Engine not available - S2/NDVI/NDWI basemaps skipped")

    def _prepare_ids_for_query(self, id_list):
        """Prepare IDs for database queries, handling both string and integer IDs."""
        return [str(id_val) for id_val in id_list]

    def reset_all_logic(self):
        """Reset all logical state, caches, and labels."""
        if self.verbose:
            print("🗑️ Resetting all labels and search results (logic)...")

        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}
        self.query_vector = None
        self.detections_with_embeddings = None
        self.vector_layer_data = None

        if self.verbose:
            print("✅ All logical data cleared!")

    def _fetch_embeddings(self, point_ids, chunk_size=None):
        """Fetch embeddings for given point IDs in chunks and cache them."""
        if chunk_size is None:
            chunk_size = DatabaseConstants.EMBEDDING_CHUNK_SIZE

        missing_ids = [pid for pid in point_ids if pid not in self.cached_embeddings]

        if not missing_ids:
            return

        if self.verbose:
            print(f"🔄 Fetching embeddings for {len(missing_ids)} points...")

        for i in range(0, len(missing_ids), chunk_size):
            chunk = missing_ids[i : i + chunk_size]

            prepared_chunk = self._prepare_ids_for_query(chunk)
            placeholders = ",".join(["?" for _ in prepared_chunk])
            query = f"""
            SELECT id, embedding
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """

            arrow_table = self.duckdb_connection.execute(
                query, prepared_chunk
            ).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()

            for _, row in chunk_df.iterrows():
                embedding = np.array(row["embedding"])
                point_id = str(row["id"])
                self.cached_embeddings[point_id] = embedding

        if self.verbose:
            print(f"✅ Cached embeddings for {len(missing_ids)} points")

    def search(self, n_neighbors: int) -> Optional[gpd.GeoDataFrame]:
        """Perform similarity search and return results as a GeoDataFrame."""
        if self.query_vector is None:
            if self.verbose:
                print(
                    "⚠️ No query vector available. Please add some positive labels first."
                )
            return None

        query_vec = self.query_vector.tolist()
        all_labeled_ids = self.pos_ids + self.neg_ids
        extra_results = min(len(all_labeled_ids), n_neighbors // 2)
        total_requested = n_neighbors + extra_results

        sql = DatabaseConstants.get_similarity_search_light_query(self.embedding_dim)
        query_params = [query_vec, total_requested]

        if self.verbose:
            print(f"🔍 Searching for {n_neighbors} similar points...")

        arrow_table = self.duckdb_connection.execute(
            sql, query_params
        ).fetch_arrow_table()
        search_results = arrow_table.select(
            ["id", "geometry_json", "geometry_wkt", "distance"]
        ).to_pandas()

        if all_labeled_ids:
            labeled_id_strings = set(str(lid) for lid in all_labeled_ids)
            mask = ~search_results["id"].astype(str).isin(labeled_id_strings)
            search_results_filtered = search_results[mask].copy()
            search_results_filtered = search_results_filtered.head(n_neighbors)
        else:
            search_results_filtered = search_results.head(n_neighbors)

        if self.verbose:
            print(f"✅ Found {len(search_results_filtered)} similar points")

        geometries = [
            shapely.wkt.loads(row["geometry_wkt"]) if row["geometry_wkt"] else None
            for _, row in search_results_filtered.iterrows()
        ]

        self.detections_with_embeddings = gpd.GeoDataFrame(
            {
                "id": search_results_filtered["id"].astype(str).values,
                "distance": search_results_filtered["distance"].values,
                "geometry": geometries,
            }
        )

        return self.detections_with_embeddings

    def find_nearest_point_id(self, lat: float, lon: float) -> Optional[str]:
        """Find the ID of the nearest point in the database to a given lat/lon."""
        clicked_point = Point(lon, lat)
        point_id = None

        if (
            self.detections_with_embeddings is not None
            and len(self.detections_with_embeddings) > 0
        ):
            distances = self.detections_with_embeddings.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()

            if distances[nearest_idx] < UIConstants.CLICK_THRESHOLD:
                nearest_detection = self.detections_with_embeddings.loc[nearest_idx]
                point_id = str(nearest_detection["id"])

        if point_id is None:
            sql = DatabaseConstants.NEAREST_POINT_LIGHT_QUERY
            arrow_table = self.duckdb_connection.execute(
                sql, [lon, lat]
            ).fetch_arrow_table()
            nearest_result = arrow_table.to_pandas()

            if nearest_result.empty:
                return None

            point_id = str(nearest_result.iloc[0]["id"])

        return point_id

    def label_point_logic(self, point_id: str):
        """Update labels for a given point ID."""
        self._fetch_embeddings([point_id])

        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)

        if self.select_val == UIConstants.POSITIVE_LABEL:
            self.pos_ids.append(point_id)
        elif self.select_val == UIConstants.NEGATIVE_LABEL:
            self.neg_ids.append(point_id)

        self.update_query_vector()

    def get_positive_layer_geojson(self) -> Dict:
        """Get GeoJSON for the positive labels layer."""
        if not self.pos_ids:
            return {"type": "FeatureCollection", "features": []}

        prepared_pos_ids = self._prepare_ids_for_query(self.pos_ids)
        placeholders = ",".join(["?" for _ in prepared_pos_ids])
        pos_query = f"""
        SELECT ST_AsGeoJSON(geometry) as geometry
        FROM geo_embeddings 
        WHERE id IN ({placeholders})
        """
        pos_results = self.duckdb_connection.execute(pos_query, prepared_pos_ids).df()
        pos_geojson = {"type": "FeatureCollection", "features": []}
        for _, row in pos_results.iterrows():
            pos_geojson["features"].append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry"]),
                    "properties": {},
                }
            )
        return pos_geojson

    def get_negative_layer_geojson(self) -> Dict:
        """Get GeoJSON for the negative labels layer."""
        if not self.neg_ids:
            return {"type": "FeatureCollection", "features": []}

        prepared_neg_ids = self._prepare_ids_for_query(self.neg_ids)
        placeholders = ",".join(["?" for _ in prepared_neg_ids])
        neg_query = f"""
        SELECT ST_AsGeoJSON(geometry) as geometry
        FROM geo_embeddings 
        WHERE id IN ({placeholders})
        """
        neg_results = self.duckdb_connection.execute(neg_query, prepared_neg_ids).df()
        neg_geojson = {"type": "FeatureCollection", "features": []}
        for _, row in neg_results.iterrows():
            neg_geojson["features"].append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry"]),
                    "properties": {},
                }
            )
        return neg_geojson

    def update_query_vector(self, skip_fetch=False):
        """Update the query vector based on current positive and negative labels."""
        if not self.pos_ids:
            self.query_vector = None
            return

        if not skip_fetch:
            self._fetch_embeddings(self.pos_ids)
            if self.neg_ids:
                self._fetch_embeddings(self.neg_ids)

        pos_embeddings = [
            self.cached_embeddings[pid]
            for pid in self.pos_ids
            if pid in self.cached_embeddings
        ]
        if not pos_embeddings:
            self.query_vector = None
            return

        pos_vec = np.mean(pos_embeddings, axis=0)
        neg_embeddings = [
            self.cached_embeddings[nid]
            for nid in self.neg_ids
            if nid in self.cached_embeddings
        ]

        if neg_embeddings:
            neg_vec = np.mean(neg_embeddings, axis=0)
        else:
            neg_vec = np.zeros_like(pos_vec)

        self.query_vector = 2 * pos_vec - neg_vec

    def save_dataset(self, b=None):
        """Save labeled points with embeddings to a GeoJSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.pos_ids and not self.neg_ids:
            if self.verbose:
                print("⚠️ No labeled points to save.")
            return

        if self.verbose:
            print("💾 Saving dataset...")

        all_labeled_ids = list(set(self.pos_ids + self.neg_ids))
        if not all_labeled_ids:
            if self.verbose:
                print("⚠️ No valid labels to save.")
            return

        prepared_labeled_ids = self._prepare_ids_for_query(all_labeled_ids)
        placeholders = ",".join(["?" for _ in prepared_labeled_ids])
        query = f"""
        SELECT id, ST_AsText(geometry) AS wkt, ST_AsGeoJSON(geometry) AS geometry_json, embedding
        FROM geo_embeddings 
        WHERE id IN ({placeholders})
        """
        results = self.duckdb_connection.execute(query, prepared_labeled_ids).df()

        if results.empty:
            if self.verbose:
                print("⚠️ Could not retrieve data for labeled points.")
            return

        features = []
        for _, row in results.iterrows():
            point_id = str(row["id"])

            if point_id in self.pos_ids:
                label = UIConstants.POSITIVE_LABEL
            elif point_id in self.neg_ids:
                label = UIConstants.NEGATIVE_LABEL
            else:
                continue

            if point_id in self.cached_embeddings:
                embedding = self.cached_embeddings[point_id]
            else:
                embedding = np.array(row["embedding"])

            feature = {
                "type": "Feature",
                "geometry": json.loads(row["geometry_json"]),
                "properties": {
                    "id": point_id,
                    "label": label,
                    "embedding": embedding.tolist(),
                },
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "positive_points": len(
                    [
                        f
                        for f in features
                        if f["properties"]["label"] == UIConstants.POSITIVE_LABEL
                    ]
                ),
                "negative_points": len(
                    [
                        f
                        for f in features
                        if f["properties"]["label"] == UIConstants.NEGATIVE_LABEL
                    ]
                ),
                "embedding_dimension": self.embedding_dim,
            },
        }

        filename = f"labeled_dataset_{timestamp}.geojson"
        try:
            with open(filename, "w") as f:
                json.dump(geojson_data, f, indent=2)

            if self.verbose:
                print(f"✅ Dataset saved successfully to {filename}")

        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving dataset: {str(e)}")

    def load_dataset_from_content(self, content: bytes, filename: str):
        """Load a dataset from uploaded file content."""
        if self.verbose:
            print(f"📂 Loading dataset from {filename}...")

        try:
            if filename.lower().endswith(".geojson"):
                geojson_data = json.loads(content.decode("utf-8"))
                self._process_geojson_data(geojson_data)
            elif filename.lower().endswith(".parquet"):
                import io

                gdf = gpd.read_parquet(io.BytesIO(content))
                self._process_geoparquet_data(gdf)
            else:
                raise ValueError(
                    "Unsupported file format. Please use .geojson or .parquet files."
                )
        except Exception as e:
            raise Exception(f"Error processing {filename}: {str(e)}")

    def _process_geojson_data(self, geojson_data):
        """Process GeoJSON data and populate labels."""
        self.pos_ids, self.neg_ids, self.cached_embeddings = [], [], {}
        for feature in geojson_data["features"]:
            point_id = str(feature["properties"]["id"])
            label = feature["properties"]["label"]
            embedding = np.array(feature["properties"]["embedding"])
            self.cached_embeddings[point_id] = embedding
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)

        self.update_query_vector()
        if self.verbose:
            print("✅ GeoJSON dataset loaded.")

    def _process_geoparquet_data(self, gdf):
        """Process GeoParquet data and populate labels."""
        self.pos_ids, self.neg_ids, self.cached_embeddings = [], [], {}
        required_cols = ["id", "label", "embedding"]
        if not all(col in gdf.columns for col in required_cols):
            raise ValueError(
                "GeoParquet file must contain 'id', 'label', and 'embedding' columns."
            )

        for _, row in gdf.iterrows():
            point_id = str(row["id"])
            label = row["label"]
            embedding = np.array(row["embedding"])
            self.cached_embeddings[point_id] = embedding
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)

        self.update_query_vector()
        if self.verbose:
            print("✅ GeoParquet dataset loaded.")

    def add_vector_layer_from_content(self, content: bytes, filename: str):
        """Add a vector layer from uploaded file content."""
        if self.verbose:
            print(f"📄 Adding vector layer from {filename}...")
        try:
            self.vector_layer_data = None
            if filename.lower().endswith(".geojson"):
                self.vector_layer_data = json.loads(content.decode("utf-8"))
            elif filename.lower().endswith(".parquet"):
                import io

                gdf = gpd.read_parquet(io.BytesIO(content))
                self.vector_layer_data = json.loads(gdf.to_json())
            else:
                raise ValueError(
                    "Unsupported file format. Please use .geojson or .parquet files."
                )
            if self.verbose:
                print("✅ Vector layer data loaded successfully.")
        except Exception as e:
            raise Exception(f"Error processing vector file {filename}: {str(e)}")

    def _construct_boundary_path(self, database_path: str) -> str:
        """Construct boundary path from database path."""
        db_filename = os.path.basename(database_path)
        db_name_with_ext = os.path.splitext(db_filename)[0]
        db_name = (
            db_name_with_ext.split("_")[0]
            if "_" in db_name_with_ext
            else db_name_with_ext
        )

        if database_path.startswith("gs://"):
            parts = database_path.split("/")
            bucket = parts[2]
            return f"gs://{bucket}/geometries/{db_name}.geojson"
        else:
            db_dir = os.path.dirname(database_path)
            parent_dir = os.path.dirname(db_dir)
            return os.path.join(parent_dir, "geometries", f"{db_name}.geojson")

    def _update_ee_boundary(self):
        """Update Earth Engine boundary based on current effective boundary path."""
        if not self.ee_available or not self.effective_boundary_path:
            self.ee_boundary = None
            return
        try:
            boundary_gdf = gpd.read_file(self.effective_boundary_path)
            self.ee_boundary = ee.Geometry(
                shapely.geometry.mapping(boundary_gdf.union_all())
            )
            if self.verbose:
                print(
                    f"🛰️ Updated Earth Engine boundary from: {self.effective_boundary_path}"
                )
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to update Earth Engine boundary: {e}")
            self.ee_boundary = None

    def _setup_boundary_and_center(self):
        """Set up boundary path and get map center coordinates."""
        boundary_path = self.config.boundary_path
        if not boundary_path and self.current_database_path:
            boundary_path = self._construct_boundary_path(self.current_database_path)

        self.effective_boundary_path = None
        if boundary_path:
            try:
                boundary_gdf = gpd.read_file(boundary_path)
                center_y, center_x = (
                    boundary_gdf.geometry.iloc[0].centroid.y,
                    boundary_gdf.geometry.iloc[0].centroid.x,
                )
                self.effective_boundary_path = boundary_path
                if self.verbose:
                    print(f"📍 Using boundary file for centering: {boundary_path}")
                return center_y, center_x
            except Exception as e:
                if self.verbose:
                    print(
                        f"⚠️  Could not load boundary file {boundary_path}: {e}. Using database centroid."
                    )

        return get_database_centroid(self.duckdb_connection, verbose=self.verbose)

    def _warm_up_gcs_database(self):
        """Warm up GCS database with initial search for better performance."""
        try:
            if self.verbose:
                print("🔧 Optimizing database connection...")

            first_point_query = "SELECT embedding FROM geo_embeddings WHERE embedding IS NOT NULL LIMIT 1"
            result = self.duckdb_connection.execute(first_point_query).fetchone()
            if not result or not result[0]:
                if self.verbose:
                    print("⚠️  No embeddings found for warm-up")
                return

            sql = DatabaseConstants.get_similarity_search_light_query(
                self.embedding_dim
            )
            self.duckdb_connection.execute(sql, [result[0], 100]).fetchall()

            if self.verbose:
                print("✅ Database optimization completed")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Database warm-up failed: {str(e)}")

    def switch_database(self, new_database_path: str):
        """Switch to a new database and re-initialize the state."""
        if new_database_path == self.current_database_path:
            return

        if self.verbose:
            print(f"🔄 Switching to database: {os.path.basename(new_database_path)}")

        try:
            # Close existing connection if owned
            if self._owns_connection and self.duckdb_connection:
                self.duckdb_connection.close()

            # Re-initialize with the new path
            self.__init__(duckdb_path=new_database_path, verbose=self.verbose)

            if self.verbose:
                print(
                    f"✅ Successfully switched to database: {os.path.basename(new_database_path)}"
                )
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch database: {str(e)}")
            # Attempt to revert to the old state - this might be tricky
            # For now, we just report the error. A more robust implementation might
            # try to re-establish the old connection.

    def close(self):
        """Clean up resources."""
        if self._owns_connection and self.duckdb_connection:
            self.duckdb_connection.close()
            if self.verbose:
                print("🔌 DuckDB connection closed.")
