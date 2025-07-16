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


def list_databases_in_directory(directory_path: str, verbose: bool = False) -> List[str]:
    """List DuckDB database files in a directory.
    
    Args:
        directory_path: Path to directory (local or GCS)
        verbose: Whether to print debug information
        
    Returns:
        List of database file paths
    """
    
    databases = []
    
    if directory_path.startswith('gs://'):
        # Handle GCS directory
        databases = _list_gcs_databases(directory_path, verbose)
    else:
        # Handle local directory
        databases = _list_local_databases(directory_path, verbose)
    
    if verbose:
        print(f"Found {len(databases)} database(s) in {directory_path}")
    
    return sorted(databases)


def _list_local_databases(directory_path: str, verbose: bool = False) -> List[str]:
    """List local DuckDB database files."""
    import glob
    
    databases = []
    
    try:
        # Look for .db files
        pattern = os.path.join(directory_path, "*.db")
        db_files = glob.glob(pattern)
        
        for db_file in db_files:
            if os.path.isfile(db_file):
                databases.append(db_file)
                if verbose:
                    print(f"  Found: {db_file}")
    except Exception as e:
        if verbose:
            print(f"Error listing local databases: {e}")
    
    return databases


def _list_gcs_databases(directory_path: str, verbose: bool = False) -> List[str]:
    """List GCS DuckDB database files.
    
    Args:
        directory_path: Path to directory (local or GCS)
        verbose: Whether to print debug information
        
    Returns:
        List of database file paths
    """
    
    databases = []
    

    import gcsfs
    fs = gcsfs.GCSFileSystem()
    
    # Remove gs:// prefix for gcsfs
    path_without_prefix = directory_path.replace('gs://', '')
    if not path_without_prefix.endswith('/'):
        path_without_prefix += '/'
    
    files = fs.glob(path_without_prefix + "*.db")
    for file_path in files:
        full_path = f"gs://{file_path}"
        databases.append(full_path)
        if verbose:
            print(f"  Found: {full_path}")



    return databases


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