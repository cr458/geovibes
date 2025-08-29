"""
Utility functions for GeoVibes.
"""

import os
from typing import Dict, Any, List
from .ui_config.constants import DatabaseConstants


def diagnose_gcs_connection(gcs_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Diagnose GCS connection issues and provide debugging information.
    
    Args:
        gcs_path: GCS path to diagnose
        verbose: Whether to print diagnostic information
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        'is_gcs_path': DatabaseConstants.is_gcs_path(gcs_path),
        'has_hmac_keys': False,
        'has_gcloud_auth': False,
        'connection_test': False,
        'error_message': None
    }
    
    if verbose:
        print(f"🔍 Diagnosing GCS connection for: {gcs_path}")
    
    # Check HMAC keys
    gcs_key_id = os.getenv('GCS_ACCESS_KEY_ID')
    gcs_secret = os.getenv('GCS_SECRET_ACCESS_KEY')
    diagnosis['has_hmac_keys'] = bool(gcs_key_id and gcs_secret)
    
    if verbose:
        if diagnosis['has_hmac_keys']:
            print("✅ HMAC keys found in environment")
        else:
            print("❌ HMAC keys not found in environment")
    
    # Check gcloud authentication
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True, timeout=10)
        diagnosis['has_gcloud_auth'] = result.returncode == 0 and 'ACTIVE' in result.stdout
        if verbose:
            if diagnosis['has_gcloud_auth']:
                print("✅ gcloud authentication available")
            else:
                print("❌ gcloud authentication not available")
    except Exception as e:
        diagnosis['has_gcloud_auth'] = False
        if verbose:
            print(f"❌ gcloud check failed: {e}")
    
    # Test connection
    if diagnosis['is_gcs_path']:
        try:
            conn = DatabaseConstants.setup_duckdb_connection(gcs_path, read_only=True)
            # Try a simple query
            result = conn.execute("SELECT 1").fetchone()
            diagnosis['connection_test'] = result is not None
            conn.close()
            if verbose:
                print("✅ Connection test successful")
        except Exception as e:
            diagnosis['connection_test'] = False
            diagnosis['error_message'] = str(e)
            if verbose:
                print(f"❌ Connection test failed: {e}")
    
    return diagnosis


def print_gcs_setup_help():
    """Print helpful GCS setup instructions."""
    print("\n🔧 GCS Setup Help:")
    print("1. Create HMAC keys at: https://console.cloud.google.com/storage/settings")
    print("2. Set environment variables:")
    print("   export GCS_ACCESS_KEY_ID='your_key_here'")
    print("   export GCS_SECRET_ACCESS_KEY='your_secret_here'")
    print("3. Or use gcloud authentication:")
    print("   gcloud auth application-default login")
    print("4. See GCS_SETUP.md for detailed instructions")
    print()


def list_databases_in_directory(
    directory_path: str, verbose: bool = False
) -> List[Dict[str, str]]:
    """List DuckDB database files in a directory and match them with FAISS index files.

    The function uses a heuristic string matching process to associate database files
    with their corresponding FAISS index files:

    1. Finds all .db files in the directory
    2. For each database file, determines the base name and applies matching logic:
       - If filename ends with "_metadata.db": strips "_metadata" suffix and looks for "{prefix}*.index"
       - Otherwise: uses full base name and looks for "{basename}*.index"

    Example matching process:
        Database file: "sentinel_usa_metadata.db"
        → Base name: "sentinel_usa_metadata" 
        → Ends with "_metadata", so prefix = "sentinel_usa"
        → Index pattern: "sentinel_usa*.index"
        → Matches: "sentinel_usa.index" or "sentinel_usa_20241201.index"

        Database file: "features.db"
        → Base name: "features"
        → Doesn't end with "_metadata"
        → Index pattern: "features*.index" 
        → Matches: "features.index" or "features_v2.index"

    Args:
        directory_path: Path to directory (local only)
        verbose: Whether to print debug information

    Returns:
        List of dictionaries with 'db_path' and 'faiss_path' keys
        
    Raises:
        ValueError: If multiple index files are found for a single database file
    """
    databases = []
    import glob

    try:
        # Look for .db files
        pattern = os.path.join(directory_path, "*.db")
        db_files = glob.glob(pattern)

        for db_file in db_files:
            if os.path.isfile(db_file):
                # Find associated FAISS index
                base_name, _ = os.path.splitext(db_file)

                # Heuristic to find index file, e.g. for something_metadata.db, look for something*.index
                if base_name.endswith("_metadata"):
                    prefix = base_name[: -len("_metadata")]
                    index_pattern = f"{prefix}*.index"
                else:
                    index_pattern = f"{base_name}*.index"

                index_files = glob.glob(index_pattern)
                if len(index_files) > 1:
                    raise ValueError(f"Multiple index files found for database '{db_file}': {index_files}")
                elif index_files:
                    databases.append({"db_path": db_file, "faiss_path": index_files[0]})
                    if verbose:
                        print(f"  Found DB: {db_file} with Index: {index_files[0]}")
                elif verbose:
                    print(
                        f"  Found DB: {db_file}, but no associated FAISS index found."
                    )

    except Exception as e:
        if verbose:
            print(f"Error listing local databases: {e}")

    if verbose:
        print(f"Found {len(databases)} database(s) in {directory_path}")

    return sorted(databases, key=lambda x: x["db_path"])


def get_database_centroid(duckdb_connection, verbose: bool = False) -> tuple:
    """Get the centroid of all points in the database.
    
    Args:
        duckdb_connection: DuckDB connection
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (latitude, longitude) for map center
    """
    try:
        # Try to get centroid from database geometries
        centroid_query = """
        SELECT 
            ST_Y(ST_Centroid(ST_Union(geometry))) as lat,
            ST_X(ST_Centroid(ST_Union(geometry))) as lon
        FROM (
            SELECT geometry 
            FROM geo_embeddings 
            LIMIT 10000
        )
        """
        
        result = duckdb_connection.execute(centroid_query).fetchone()
        
        if result and result[0] is not None and result[1] is not None:
            lat, lon = result[0], result[1]
            if verbose:
                print(f"📍 Database centroid: {lat:.4f}, {lon:.4f}")
            return lat, lon
        else:
            # Fallback: get average of individual point coordinates
            avg_query = """
            SELECT 
                AVG(ST_Y(geometry)) as avg_lat,
                AVG(ST_X(geometry)) as avg_lon
            FROM geo_embeddings
            LIMIT 1000
            """
            
            result = duckdb_connection.execute(avg_query).fetchone()
            if result and result[0] is not None and result[1] is not None:
                lat, lon = result[0], result[1]
                if verbose:
                    print(f"📍 Database center (avg): {lat:.4f}, {lon:.4f}")
                return lat, lon
            
    except Exception as e:
        if verbose:
            print(f"⚠️  Could not get database centroid: {e}")
    
    # Ultimate fallback: center of world
    if verbose:
        print("📍 Using default center (0, 0)")
    return 0.0, 0.0 