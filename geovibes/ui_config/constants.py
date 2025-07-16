"""
Constants for the GeoVibes application.
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class UIConstants:
    """UI-related constants."""
    
    # Colors
    POS_COLOR = '#0072B2'       # Blue
    NEG_COLOR = '#D55E00'       # Orange  
    NEUTRAL_COLOR = '#999999'   # Grey
    REGION_COLOR = '#FAFAFA'    # Light gray
    SEARCH_COLOR = '#ffe014'    # Yellow
    DRAW_COLOR = '#6be5c3'      # Teal
    
    # Map settings
    DEFAULT_ZOOM = 7
    DEFAULT_HEIGHT = '700px'
    PANEL_WIDTH = '200px'
    
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
    BUTTON_HEIGHT = '40px'
    RESET_BUTTON_HEIGHT = '35px'
    COLLAPSE_BUTTON_SIZE = '25px'
    
    # Click threshold for point selection
    CLICK_THRESHOLD = 0.001
    
    # Label values
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    ERASE_LABEL = -100
    
    # Distance-based color mapping
    DISTANCE_COLORS = {
        'HIGH_SIMILARITY': '#00ff00',    # Green (low distance, high similarity)
        'MEDIUM_SIMILARITY': '#ffff00',  # Yellow (medium distance)
        'LOW_SIMILARITY': '#ff4444',     # Red (high distance, low similarity)
        'DEFAULT': '#ffe014'             # Default yellow
    }
    
    @staticmethod
    def distance_to_color(distance: float, min_dist: float, max_dist: float) -> str:
        """Convert distance value to hex color using a green-yellow-red gradient.
        
        Args:
            distance: Distance value to convert.
            min_dist: Minimum distance in the dataset.
            max_dist: Maximum distance in the dataset.
            
        Returns:
            Hex color string representing similarity (green=high, red=low).
        """
        if max_dist == min_dist:
            return UIConstants.DISTANCE_COLORS['MEDIUM_SIMILARITY']
        
        # Normalize distance to 0-1 range
        normalized = (distance - min_dist) / (max_dist - min_dist)
        normalized = np.clip(normalized, 0, 1)
        
        # Create gradient: green (0) -> yellow (0.5) -> red (1)
        if normalized <= 0.5:
            # Green to yellow
            r = int(255 * (normalized * 2))
            g = 255
            b = 0
        else:
            # Yellow to red  
            r = 255
            g = int(255 * (1 - (normalized - 0.5) * 2))
            b = 0
        
        return f'#{r:02x}{g:02x}{b:02x}'


class BasemapConfig:
    """Basemap configuration and tile URLs."""
    
    MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')
    
    # Attribution strings
    MAPTILER_ATTRIBUTION = (
        '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> '
        '<a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>'
    )
    
    # Base tile URLs
    BASEMAP_TILES = {
        'MAPTILER': f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
        'HUTCH_TILE': 'https://tiles.earthindex.ai/v1/tiles/sentinel2-yearly-mosaics/2024-01-01/2025-01-01/rgb/{z}/{x}/{y}.webp',
        'GOOGLE_HYBRID': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    }
    
    # Earth Engine basemap visualization parameters
    S2_RGB_VIS_PARAMS = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2']
    }
    
    NDVI_VIS_PARAMS = {
        'min': -0.1,
        'max': 1.0,
        'palette': ['red', 'yellow', 'green']
    }
    
    NDWI_VIS_PARAMS = {
        'min': -0.5,
        'max': 0.5,
        'palette': ['brown', 'white', 'blue']
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
    
    @classmethod
    def get_memory_setup_queries(cls):
        """Get memory configuration queries."""
        return [
            f"SET memory_limit='{cls.MEMORY_LIMIT}'",
            f"SET max_memory='{cls.MAX_MEMORY}'",
            f"SET temp_directory='{cls.TEMP_DIRECTORY}'"
        ]
    
    @classmethod
    def get_extension_setup_queries(cls, duckdb_path: str):
        """Get extension setup queries based on database path.
        
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
        return path.startswith('gs://')
    
    @classmethod
    def setup_duckdb_connection(cls, duckdb_path: str, read_only: bool = True):
        """Set up DuckDB connection for both local and GCS databases.
        
        Args:
            duckdb_path: Path to DuckDB database (local or GCS)
            read_only: Whether to open in read-only mode
            
        Returns:
            DuckDB connection object
        """
        import duckdb
        
        if cls.is_gcs_path(duckdb_path):
            # For GCS paths, create in-memory connection and attach remote database
            conn = duckdb.connect(':memory:')
            
            # Install and load httpfs extension
            conn.execute(cls.HTTPFS_EXTENSION_SETUP_QUERY)
            
            # Set up GCS authentication if credentials are available
            cls._setup_gcs_auth(conn)
            
            # Attach the remote database
            attach_query = f"ATTACH '{duckdb_path}' AS remote_db (READ_ONLY)"
            conn.execute(attach_query)
            
            # Create view to map table name for transparent access
            conn.execute("CREATE VIEW geo_embeddings AS SELECT * FROM remote_db.geo_embeddings")
            
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
        gcs_key_id = os.getenv('GCS_ACCESS_KEY_ID')
        gcs_secret = os.getenv('GCS_SECRET_ACCESS_KEY')
        
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
    MEMORY_LIMIT = '12GB'
    MAX_MEMORY = '12GB'
    TEMP_DIRECTORY = '/tmp'
    
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
    def detect_embedding_dimension_from_parquet(parquet_path: str, embedding_column: str = 'embedding') -> int:
        """Detect embedding dimension from parquet file.
        
        Args:
            parquet_path: Path to parquet file
            embedding_column: Name of embedding column (default: 'embedding')
            
        Returns:
            int: Embedding dimension
            
        Raises:
            ValueError: If no embeddings found or dimension cannot be detected
        """
        try:
            import pandas as pd
            
            # Read just the first row
            df = pd.read_parquet(parquet_path, nrows=1)
            
            if embedding_column not in df.columns:
                raise ValueError(f"Embedding column '{embedding_column}' not found in parquet file")
            
            embedding = df[embedding_column].iloc[0]
            if hasattr(embedding, '__len__'):
                return len(embedding)
            else:
                raise ValueError(f"Embedding in column '{embedding_column}' is not array-like")
                
        except Exception as e:
            raise ValueError(f"Could not detect embedding dimension from parquet: {e}")
    
    @staticmethod
    def get_similarity_search_query(embedding_dim: int) -> str:
        """Generate similarity search query with embeddings for given dimension.
        
        Args:
            embedding_dim: Dimension of the embeddings
            
        Returns:
            str: SQL query string
        """
        return f"""
        WITH query(vec) AS (SELECT CAST(? AS FLOAT[{embedding_dim}]))
        SELECT  g.id,
                g.embedding,
                ST_AsGeoJSON(g.geometry) AS geometry_json,
                ST_AsText(g.geometry) AS geometry_wkt,
                array_distance(g.embedding, q.vec) AS distance
        FROM    geo_embeddings AS g, query AS q
        ORDER BY distance
        LIMIT ?;
        """
    
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
    
    # Legacy constants for backward compatibility (deprecated)
    @property
    def EMBEDDING_DIM(self):
        """Deprecated: Use detect_embedding_dimension() instead."""
        import warnings
        warnings.warn(
            "EMBEDDING_DIM is deprecated. Use detect_embedding_dimension() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return 1000  # Default fallback
    
    @property 
    def SIMILARITY_SEARCH_QUERY(self):
        """Deprecated: Use get_similarity_search_query() instead."""
        import warnings
        warnings.warn(
            "SIMILARITY_SEARCH_QUERY is deprecated. Use get_similarity_search_query() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_similarity_search_query(1000)
    
    @property
    def SIMILARITY_SEARCH_LIGHT_QUERY(self):
        """Deprecated: Use get_similarity_search_light_query() instead."""
        import warnings
        warnings.warn(
            "SIMILARITY_SEARCH_LIGHT_QUERY is deprecated. Use get_similarity_search_light_query() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_similarity_search_light_query(1000)

    # Nearest point query without embedding (memory-efficient)
    NEAREST_POINT_LIGHT_QUERY = """
    SELECT  g.id,
            ST_AsText(g.geometry) AS wkt,
            ST_Distance(geometry, ST_Point(?, ?)) AS dist_m
    FROM    geo_embeddings g
    ORDER BY dist_m
    LIMIT   1
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
            'color': UIConstants.REGION_COLOR,
            'opacity': 1,
            'fillOpacity': 0,
            'weight': 1
        }
    
    @classmethod
    def get_point_style(cls, color):
        """Get point layer style for given color."""
        return {
            'color': color,
            'radius': UIConstants.POINT_RADIUS,
            'fillColor': color,
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.POINT_WEIGHT
        }
    
    @classmethod
    def get_erase_style(cls):
        """Get erase layer style."""
        return {
            'color': 'white',
            'radius': UIConstants.POINT_RADIUS,
            'fillColor': '#000000',
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.POINT_WEIGHT
        }
    
    @classmethod
    def get_search_style(cls):
        """Get search results layer style."""
        return {
            'color': 'black',
            'radius': UIConstants.SEARCH_POINT_RADIUS,
            'fillColor': UIConstants.SEARCH_COLOR,
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.SEARCH_POINT_WEIGHT
        }
    
    @classmethod
    def get_search_hover_style(cls):
        """Get search results hover style."""
        return {
            'fillColor': UIConstants.SEARCH_COLOR,
            'fillOpacity': 0.5
        }
    
    @classmethod
    def get_draw_options(cls):
        """Get draw control options."""
        return {
            "shapeOptions": {
                "color": UIConstants.DRAW_COLOR, 
                "fillOpacity": 0.5
            }
        } 


