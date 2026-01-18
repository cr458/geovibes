"""FAISS index and geometry caching for remote files.

Downloads FAISS indexes and geometry caches from S3 and caches them locally for fast access.
"""

import hashlib
import logging
from pathlib import Path

import faiss
import fsspec
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FaissCache:
    """Download and cache FAISS indexes and geometry files locally."""

    def __init__(self, cache_dir: Path = None):
        """Initialize cache with optional custom directory.

        Args:
            cache_dir: Base directory for cache.
                       Defaults to ~/.cache/geovibes/
        """
        base_dir = cache_dir or Path.home() / ".cache" / "geovibes"
        self.cache_dir = base_dir / "faiss"
        self.geometry_cache_dir = base_dir / "geometry"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.geometry_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate deterministic cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL."""
        cache_key = self._get_cache_key(url)
        return self.cache_dir / f"{cache_key}.faiss"

    def get_index(self, remote_url: str, show_progress: bool = True) -> faiss.Index:
        """Load index from cache, downloading if necessary.

        Args:
            remote_url: S3 or other fsspec-compatible URL to the FAISS index.
            show_progress: Whether to show download progress bar.

        Returns:
            Loaded FAISS index.

        Raises:
            RuntimeError: If index is corrupted and cannot be loaded.
            FileNotFoundError: If remote URL doesn't exist.
        """
        cache_path = self._get_cache_path(remote_url)

        # Try loading from cache
        if cache_path.exists():
            try:
                return self._load_index(cache_path)
            except Exception as e:
                logger.warning(f"Cached index corrupted, will re-download: {e}")
                cache_path.unlink(missing_ok=True)

        # Download to cache
        self._download(remote_url, cache_path, show_progress)

        # Load and validate
        return self._load_index(cache_path)

    def _load_index(self, path: Path) -> faiss.Index:
        """Load and validate a FAISS index from disk."""
        try:
            # Try memory-mapped loading for efficiency
            index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
            # Validate by accessing properties
            _ = index.ntotal
            _ = index.d
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {path}: {e}")

    def _download(self, url: str, dest: Path, show_progress: bool):
        """Download file with progress and integrity checking."""
        temp_path = dest.with_suffix(".tmp")

        try:
            with fsspec.open(url, "rb") as src:
                size = getattr(src, "size", None)

                with open(temp_path, "wb") as dst:
                    pbar = None
                    if show_progress and size:
                        pbar = tqdm(
                            total=size,
                            unit="B",
                            unit_scale=True,
                            desc="Downloading FAISS index",
                        )

                    try:
                        while True:
                            chunk = src.read(8 * 1024 * 1024)  # 8MB chunks
                            if not chunk:
                                break
                            dst.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    finally:
                        if pbar:
                            pbar.close()

            # Atomic rename
            temp_path.rename(dest)
            logger.info(f"Downloaded FAISS index to {dest}")

        except Exception as e:
            # Clean up partial download
            temp_path.unlink(missing_ok=True)
            if "404" in str(e) or "not found" in str(e).lower():
                raise FileNotFoundError(f"FAISS index not found: {url}")
            raise

    def is_cached(self, remote_url: str) -> bool:
        """Check if an index is already cached."""
        return self._get_cache_path(remote_url).exists()

    def clear_cache(self):
        """Remove all cached indexes."""
        for f in self.cache_dir.glob("*.faiss"):
            f.unlink()
        logger.info("Cleared FAISS cache")

    # --- Geometry Cache Methods ---

    def _get_geometry_cache_path(self, url: str) -> Path:
        """Get the cache file path for a geometry cache URL."""
        cache_key = self._get_cache_key(url)
        return self.geometry_cache_dir / f"{cache_key}.parquet"

    def get_geometry_cache(self, remote_url: str, show_progress: bool = True) -> Path:
        """Download geometry cache and return local path.

        Args:
            remote_url: S3 or other fsspec-compatible URL to the geometry Parquet file.
            show_progress: Whether to show download progress bar.

        Returns:
            Path to the local cached geometry file.

        Raises:
            FileNotFoundError: If remote URL doesn't exist.
        """
        cache_path = self._get_geometry_cache_path(remote_url)

        # Return cached path if exists
        if cache_path.exists():
            logger.info(f"Using cached geometry file: {cache_path}")
            return cache_path

        # Download to cache
        self._download_geometry(remote_url, cache_path, show_progress)
        return cache_path

    def _download_geometry(self, url: str, dest: Path, show_progress: bool):
        """Download geometry cache file with progress."""
        temp_path = dest.with_suffix(".tmp")

        try:
            with fsspec.open(url, "rb") as src:
                size = getattr(src, "size", None)

                with open(temp_path, "wb") as dst:
                    pbar = None
                    if show_progress and size:
                        pbar = tqdm(
                            total=size,
                            unit="B",
                            unit_scale=True,
                            desc="Downloading geometry cache",
                        )

                    try:
                        while True:
                            chunk = src.read(8 * 1024 * 1024)  # 8MB chunks
                            if not chunk:
                                break
                            dst.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    finally:
                        if pbar:
                            pbar.close()

            # Atomic rename
            temp_path.rename(dest)
            logger.info(f"Downloaded geometry cache to {dest}")

        except Exception as e:
            # Clean up partial download
            temp_path.unlink(missing_ok=True)
            if "404" in str(e) or "not found" in str(e).lower():
                raise FileNotFoundError(f"Geometry cache not found: {url}")
            raise

    def is_geometry_cached(self, remote_url: str) -> bool:
        """Check if a geometry cache is already downloaded."""
        return self._get_geometry_cache_path(remote_url).exists()

    def clear_geometry_cache(self):
        """Remove all cached geometry files."""
        for f in self.geometry_cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info("Cleared geometry cache")
