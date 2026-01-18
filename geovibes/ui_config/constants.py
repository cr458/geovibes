"""
Constants for the GeoVibes application.
"""

import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Ensure .env is loaded when Voila runs from a temp directory
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")


class UIConstants:
    """UI-related constants."""

    # Colors
    POS_COLOR = "#0072B2"  # Blue
    NEG_COLOR = "#D55E00"  # Orange
    NEUTRAL_COLOR = "#999999"  # Grey
    REGION_COLOR = "#FAFAFA"  # Light gray
    SEARCH_COLOR = "#ffe014"  # Yellow
    DRAW_COLOR = "#6be5c3"  # Teal

    # Map settings
    DEFAULT_ZOOM = 7
    DEFAULT_HEIGHT = "780px"
    PANEL_WIDTH = "280px"

    # Search settings
    DEFAULT_NEIGHBORS = 1000
    MIN_NEIGHBORS = 100
    MAX_NEIGHBORS = 25000
    NEIGHBORS_STEP = 100

    # Point styles
    POINT_RADIUS = 4
    SEARCH_POINT_RADIUS = 3
    POINT_OPACITY = 1
    POINT_FILL_OPACITY = 0.7
    POINT_WEIGHT = 2
    SEARCH_POINT_WEIGHT = 1

    # UI dimensions
    BUTTON_HEIGHT = "40px"
    RESET_BUTTON_HEIGHT = "35px"
    COLLAPSE_BUTTON_SIZE = "25px"

    # Label values
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    ERASE_LABEL = -100

    SEARCH_COLORMAP = os.getenv("GEOVIBES_SEARCH_COLORMAP", "turbo")
    _COLORMAP_CACHE: dict[str, dict[str, object]] = {}

    @classmethod
    def _get_colormap_bundle(cls) -> dict[str, object]:
        requested = (
            os.getenv("GEOVIBES_SEARCH_COLORMAP", cls.SEARCH_COLORMAP)
            or cls.SEARCH_COLORMAP
        )
        if requested not in cls._COLORMAP_CACHE:
            try:
                cmap = mpl_colormaps.get_cmap(requested)
                cmap_name = requested
            except ValueError:
                cmap = mpl_colormaps.get_cmap("viridis")
                cmap_name = "viridis"

            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack([gradient, gradient])
            fig = Figure(figsize=(2.4, 0.3))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.imshow(gradient, aspect="auto", cmap=cmap)
            ax.set_axis_off()
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            colorbar_data = base64.b64encode(buf.read()).decode("ascii")
            cls._COLORMAP_CACHE[requested] = {
                "cmap": cmap,
                "colorbar": colorbar_data,
                "resolved_name": cmap_name,
            }
            cls.SEARCH_COLORMAP = cmap_name
        else:
            cls.SEARCH_COLORMAP = cls._COLORMAP_CACHE[requested]["resolved_name"]  # type: ignore[index]

        return cls._COLORMAP_CACHE[requested]

    @classmethod
    def _color_from_fraction(cls, fraction: float) -> str:
        bundle = cls._get_colormap_bundle()
        cmap = bundle["cmap"]  # type: ignore[index]
        r, g, b, _ = cmap(np.clip(fraction, 0.0, 1.0))
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    @staticmethod
    def distance_to_color(
        distance: float,
        min_dist: float,
        max_dist: float,
        highlight_cutoff: float | None = None,
    ) -> str:
        return UIConstants._distance_to_color(
            distance, min_dist, max_dist, highlight_cutoff
        )

    @classmethod
    def _distance_to_color(
        cls,
        distance: float,
        min_dist: float,
        max_dist: float,
        highlight_cutoff: float | None = None,
    ) -> str:
        """Convert distance value to hex color using the configured matplotlib colormap.

        Args:
            distance: Distance value to convert.
            min_dist: Minimum distance in the dataset.
            max_dist: Maximum distance in the dataset.
            highlight_cutoff: Optional distance threshold marking the lower distances that should
                map to the warm end of the colormap.

        Returns:
            Hex color string representing similarity (yellow=high similarity, dark blue=low).
        """
        if max_dist == min_dist:
            return cls._color_from_fraction(0.5)

        # Normalize distance to 0-1 range
        if highlight_cutoff is not None and highlight_cutoff > min_dist:
            cutoff = min(highlight_cutoff, max_dist)
            if cutoff <= min_dist:
                highlight_cutoff = None
        if highlight_cutoff is not None and highlight_cutoff > min_dist:
            cutoff = min(highlight_cutoff, max_dist)
            if distance <= cutoff:
                span = cutoff - min_dist if cutoff > min_dist else 1.0
                local = (distance - min_dist) / span
                local = np.clip(local, 0, 1)
                color_fraction = 0.7 + (1.0 - local) * 0.3
            else:
                span = max_dist - cutoff if max_dist > cutoff else 1.0
                local = (distance - cutoff) / span
                local = np.clip(local, 0, 1)
                color_fraction = max(0.0, 0.7 - local * 0.7)
        else:
            normalized = (distance - min_dist) / (max_dist - min_dist)
            normalized = np.clip(normalized, 0, 1)
            color_fraction = 1.0 - normalized
        return cls._color_from_fraction(color_fraction)

    @classmethod
    def similarity_colorbar_data_uri(cls) -> str:
        bundle = cls._get_colormap_bundle()
        return bundle["colorbar"]  # type: ignore[index]

    @classmethod
    def similarity_legend_html(cls) -> str:
        colorbar = cls.similarity_colorbar_data_uri()
        low_color = cls._color_from_fraction(0.0)
        high_color = cls._color_from_fraction(1.0)
        return (
            "<div style='display:flex; align-items:center; gap:8px; margin-top:6px;'>"
            "<strong>Similarity:</strong>"
            f"<span style='color:{low_color}; font-weight:bold;'>Least similar</span>"
            f"<img src='data:image/png;base64,{colorbar}' alt='similarity colorbar' "
            "style='height:12px; flex:1 1 auto;'/>"
            f"<span style='color:{high_color}; font-weight:bold;'>Most similar</span>"
            "</div>"
        )


class BasemapConfig:
    """Basemap configuration and tile URLs."""

    MAPTILER_API_KEY = os.getenv("MAPTILER_API_KEY")

    # Attribution strings
    MAPTILER_ATTRIBUTION = (
        '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> '
        '<a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>'
    )

    # Base tile URLs
    BASEMAP_TILES = {
        "MAPTILER": f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
        "HUTCH_TILE": "https://tiles.earthindex.ai/v1/tiles/sentinel2-yearly-mosaics/2024-01-01/2025-01-01/rgb/{z}/{x}/{y}.webp",
        "GOOGLE_HYBRID": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    }


class DatabaseConstants:
    """Database-related constants."""

    EXTENSION_SETUP_QUERY = """
    INSTALL spatial;
    LOAD spatial;
    """

    HTTPFS_EXTENSION_SETUP_QUERY = """
    INSTALL httpfs;
    LOAD httpfs;
    """

    # Base URL for remote database discovery (scans all regions/dates)
    DEFAULT_SEARCH_BASE_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/"

    # S3 endpoint for Source Cooperative
    SOURCE_COOP_ENDPOINT = "https://data.source.coop"
    SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"

    @classmethod
    def get_memory_setup_queries(cls):
        """Get memory configuration queries."""
        return [
            f"SET memory_limit='{cls.MEMORY_LIMIT}'",
            f"SET max_memory='{cls.MAX_MEMORY}'",
            f"SET temp_directory='{cls.TEMP_DIRECTORY}'",
            # Disable progress bar to prevent Jupyter crashes with UTINYINT[] arrays
            "SET enable_progress_bar=false",
            "SET enable_profiling='no_output'",
        ]

    @classmethod
    def get_extension_setup_queries(cls, duckdb_path: str):
        """Get extension setup queries based on database path and index type.

        Args:
            duckdb_path: Path to DuckDB database (local or GCS)

        Returns:
            List of SQL queries to set up required extensions
        """
        queries = [cls.EXTENSION_SETUP_QUERY]

        # Add httpfs extension if using GCS
        if cls.is_gcs_path(duckdb_path):
            queries.insert(0, cls.HTTPFS_EXTENSION_SETUP_QUERY)

        return queries

    @classmethod
    def is_gcs_path(cls, path: str) -> bool:
        """Check if path is a Google Cloud Storage path.

        Args:
            path: Path to check

        Returns:
            True if path is GCS URL, False otherwise
        """
        return path.startswith("gs://")

    @classmethod
    def is_s3_path(cls, path: str) -> bool:
        """Check if path is an S3 path.

        Args:
            path: Path to check

        Returns:
            True if path is S3 URL, False otherwise
        """
        return path.startswith("s3://")

    @classmethod
    def setup_duckdb_connection(cls, duckdb_path: str, read_only: bool = True):
        """Set up DuckDB connection for local, GCS, or S3 databases.

        Args:
            duckdb_path: Path to DuckDB database (local, GCS, or S3)
            read_only: Whether to open in read-only mode

        Returns:
            DuckDB connection object
        """
        import duckdb

        if cls.is_gcs_path(duckdb_path):
            # For GCS paths, create in-memory connection and attach remote database
            conn = duckdb.connect(":memory:")

            # Install and load httpfs extension
            conn.execute(cls.HTTPFS_EXTENSION_SETUP_QUERY)

            # Set up GCS authentication if credentials are available
            cls._setup_gcs_auth(conn)

            # Attach the remote database
            attach_query = f"ATTACH '{duckdb_path}' AS remote_db (READ_ONLY)"
            conn.execute(attach_query)

            # Create view to map table name for transparent access
            conn.execute(
                "CREATE VIEW geo_embeddings AS SELECT * FROM remote_db.geo_embeddings"
            )

            return conn
        elif cls.is_s3_path(duckdb_path):
            # For S3 paths, create in-memory connection and attach remote database
            conn = duckdb.connect(":memory:")

            # Install and load required extensions
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("INSTALL spatial; LOAD spatial;")
            conn.execute("INSTALL aws; LOAD aws;")

            # Load AWS credentials from environment/config
            conn.execute("CALL load_aws_credentials();")

            # Attach the remote database
            attach_query = f"ATTACH '{duckdb_path}' AS remote_db (READ_ONLY)"
            conn.execute(attach_query)

            # Create view to map table name for transparent access
            conn.execute(
                "CREATE VIEW geo_embeddings AS SELECT * FROM remote_db.geo_embeddings"
            )

            return conn
        else:
            # For local paths, use direct connection
            return duckdb.connect(duckdb_path, read_only=read_only)

    @classmethod
    def _setup_gcs_auth(cls, conn):
        """Set up GCS authentication using environment variables.

        Args:
            conn: DuckDB connection
        """
        import os

        # Try to get HMAC keys from environment variables
        gcs_key_id = os.getenv("GCS_ACCESS_KEY_ID")
        gcs_secret = os.getenv("GCS_SECRET_ACCESS_KEY")

        if gcs_key_id and gcs_secret:
            # Create secret using HMAC keys
            conn.execute(f"""
                CREATE SECRET (
                    TYPE gcs,
                    KEY_ID '{gcs_key_id}',
                    SECRET '{gcs_secret}'
                );
            """)
        else:
            # Try to use default Google Cloud authentication
            # This will work if running on GCP or if gcloud is configured
            try:
                conn.execute("""
                    CREATE SECRET (
                        TYPE gcs,
                        PROVIDER credential_chain
                    );
                """)
            except Exception:
                # If no authentication available, continue without auth
                # This may work for public buckets or if other auth is configured
                pass

    # Memory configuration
    MEMORY_LIMIT = "24GB"
    MAX_MEMORY = "24GB"
    TEMP_DIRECTORY = "/tmp"

    # Chunk size for embedding fetching to avoid memory issues
    EMBEDDING_CHUNK_SIZE = 10000

    @staticmethod
    def detect_embedding_dimension(duckdb_connection) -> int:
        """Detect embedding dimension from first row of database.

        Args:
            duckdb_connection: Active DuckDB connection

        Returns:
            int: Embedding dimension

        Raises:
            ValueError: If no embeddings found or dimension cannot be detected
        """
        try:
            result = duckdb_connection.execute(
                "SELECT embedding FROM geo_embeddings LIMIT 1"
            ).fetchone()

            if result and result[0]:
                return len(result[0])
            else:
                raise ValueError("No embeddings found in database")
        except Exception as e:
            raise ValueError(f"Could not detect embedding dimension: {e}")

    @staticmethod
    def get_similarity_search_light_query(embedding_dim: int) -> str:
        """Generate lightweight similarity search query for given dimension.

        Args:
            embedding_dim: Dimension of the embeddings

        Returns:
            str: SQL query string
        """
        return f"""
        SELECT  g.id,
                ST_AsGeoJSON(g.geometry) AS geometry_json,
                ST_AsText(g.geometry)  AS geometry_wkt,
                distance
        FROM (
            SELECT  g.id,
                    g.geometry,
                    g.embedding <-> CAST(? AS FLOAT[{embedding_dim}]) AS distance
            FROM    geo_embeddings g
            WHERE   g.embedding IS NOT NULL
            ORDER BY distance
            LIMIT ?
        ) g;
        """

    # Original nearest point query with embedding (kept for backward compatibility)
    NEAREST_POINT_QUERY = """
    SELECT  g.id,
            ST_AsText(g.geometry) AS wkt,
            ST_Distance(geometry, ST_Point(?, ?)) AS dist_m,
            g.embedding
    FROM    geo_embeddings g
    ORDER BY dist_m
    LIMIT   1
    """


class LayerStyles:
    """Map layer styling constants."""

    @classmethod
    def get_region_style(cls):
        """Get region boundary style."""
        return {
            "color": UIConstants.REGION_COLOR,
            "opacity": 1,
            "fillOpacity": 0,
            "weight": 1,
        }

    @classmethod
    def get_point_style(cls, color):
        """Get point layer style for given color."""
        return {
            "color": color,
            "radius": UIConstants.POINT_RADIUS,
            "fillColor": color,
            "opacity": UIConstants.POINT_OPACITY,
            "fillOpacity": UIConstants.POINT_FILL_OPACITY,
            "weight": UIConstants.POINT_WEIGHT,
        }

    @classmethod
    def get_erase_style(cls):
        """Get erase layer style."""
        return {
            "color": "white",
            "radius": UIConstants.POINT_RADIUS,
            "fillColor": "#000000",
            "opacity": UIConstants.POINT_OPACITY,
            "fillOpacity": UIConstants.POINT_FILL_OPACITY,
            "weight": UIConstants.POINT_WEIGHT,
        }

    @classmethod
    def get_search_style(cls):
        """Get search results layer style."""
        return {
            "color": "black",
            "radius": UIConstants.SEARCH_POINT_RADIUS,
            "fillColor": UIConstants.SEARCH_COLOR,
            "opacity": UIConstants.POINT_OPACITY,
            "fillOpacity": UIConstants.POINT_FILL_OPACITY,
            "weight": UIConstants.SEARCH_POINT_WEIGHT,
        }

    @classmethod
    def get_search_hover_style(cls):
        """Get search results hover style."""
        return {"fillColor": UIConstants.SEARCH_COLOR, "fillOpacity": 0.5}

    @classmethod
    def get_draw_options(cls):
        """Get draw control options."""
        return {"shapeOptions": {"color": UIConstants.DRAW_COLOR, "fillOpacity": 0.5}}

    @classmethod
    def get_detection_style(cls):
        """Get base detection layer style."""
        return {
            "color": "#00FF00",
            "weight": 2,
            "opacity": 0.8,
            "fillOpacity": 0.3,
        }

    @classmethod
    def probability_to_color(cls, probability: float) -> str:
        """Convert probability (0-1) to hex color using the configured colormap.

        Uses the same colormap as distance_to_color for visual consistency.
        Higher probability = brighter/warmer color (like lower distance).

        Args:
            probability: Value between 0 and 1

        Returns:
            Hex color string
        """
        return UIConstants._color_from_fraction(probability)
