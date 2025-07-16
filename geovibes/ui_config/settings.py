"""
Configuration management for GeoVibes.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeoVibesConfig:
    """Configuration for GeoVibes application."""
    
    duckdb_path: Optional[str] = None
    duckdb_directory: Optional[str] = None
    boundary_path: Optional[str] = None
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    gcp_project: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'GeoVibesConfig':
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            GeoVibesConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Check for required database path (either single file or directory)
        has_duckdb_path = 'duckdb_path' in config_data
        has_duckdb_directory = 'duckdb_directory' in config_data
        
        if not has_duckdb_path and not has_duckdb_directory:
            raise ValueError("Either 'duckdb_path' or 'duckdb_directory' must be provided")
        
        # Validate other required fields
        required_fields = ['start_date', 'end_date']
        missing_fields = [field for field in required_fields if field not in config_data]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Create config instance with all available fields
        config_dict = {
            'start_date': config_data['start_date'],
            'end_date': config_data['end_date']
        }
        
        # Add optional fields if present
        if has_duckdb_path:
            config_dict['duckdb_path'] = config_data['duckdb_path']
        if has_duckdb_directory:
            config_dict['duckdb_directory'] = config_data['duckdb_directory']
        if 'boundary_path' in config_data:
            config_dict['boundary_path'] = config_data['boundary_path']
        if 'gcp_project' in config_data:
            config_dict['gcp_project'] = config_data['gcp_project']
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GeoVibesConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            GeoVibesConfig instance
        """
        return cls(
            duckdb_path=config_dict['duckdb_path'],
            boundary_path=config_dict['boundary_path'],
            start_date=config_dict['start_date'],
            end_date=config_dict['end_date'],
            gcp_project=config_dict.get('gcp_project')
        )
    
    def _path_exists(self, path: str) -> bool:
        """Check if a path exists, supporting both local and GCS paths.
        
        Args:
            path: File path (local or GCS URL)
            
        Returns:
            True if path exists, False otherwise
        """
        if path.startswith('gs://'):
            # For GCS paths, validate URL format
            if not self._is_valid_gcs_url(path):
                return False
            # For now, we assume valid GCS URLs exist
            # In production, you could use gcsfs to check existence:
            # try:
            #     import gcsfs
            #     fs = gcsfs.GCSFileSystem()
            #     return fs.exists(path)
            # except ImportError:
            #     pass
            return True
        else:
            # For local paths, use standard os.path.exists
            return os.path.exists(path)
    
    def _is_valid_gcs_url(self, url: str) -> bool:
        """Validate GCS URL format.
        
        Args:
            url: GCS URL to validate
            
        Returns:
            True if URL format is valid, False otherwise
        """
        import re
        # Basic GCS URL pattern: gs://bucket-name/path/to/file
        pattern = r'^gs://[a-z0-9]([a-z0-9\-._]{0,61}[a-z0-9])?(/.*)?$'
        return bool(re.match(pattern, url))
    
    def validate(self) -> None:
        """Validate the configuration.
        
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If configuration values are invalid
        """
        # Check that either duckdb_path or duckdb_directory is provided
        if not self.duckdb_path and not self.duckdb_directory:
            raise ValueError("Either duckdb_path or duckdb_directory must be provided")
        
        # Check paths exist if provided
        if self.boundary_path and not self._path_exists(self.boundary_path):
            raise FileNotFoundError(f"Boundary file not found: {self.boundary_path}")
        
        if self.duckdb_path and not self._path_exists(self.duckdb_path):
            raise FileNotFoundError(f"DuckDB file not found: {self.duckdb_path}")
        
        if self.duckdb_directory and not self._path_exists(self.duckdb_directory):
            raise FileNotFoundError(f"DuckDB directory not found: {self.duckdb_directory}")
        
        # Validate date format (basic check)
        import datetime
        try:
            datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        # Check that start_date is before end_date
        start = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            'duckdb_path': self.duckdb_path,
            'boundary_path': self.boundary_path,
            'start_date': self.start_date,
            'end_date': self.end_date
        }
        if self.gcp_project:
            config_dict['gcp_project'] = self.gcp_project
        return config_dict 