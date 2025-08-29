"""Map tile fetching utilities."""

import math

import requests
import tenacity

from .ui_config import BasemapConfig


def deg2num(lat_deg: float, lon_deg: float, zoom: int):
    """Convert degrees to tile numbers.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        zoom: Zoom level.

    Returns:
        A tuple of (xtile, ytile).
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
)
def get_map_image(source: str, lon: float, lat: float, zoom: int = 17) -> bytes:
    """Fetch a map tile image for a given coordinate and source.

    Args:
        source: The tile source (e.g., 'MAPTILER', 'GOOGLE_HYBRID').
        lon: Longitude of the point.
        lat: Latitude of the point.
        zoom: The zoom level for the tile.

    Returns:
        The image content as bytes.
    
    Raises:
        ValueError: If the source is not a valid XYZ tile source.
        requests.exceptions.RequestException: If the image download fails.
    """
    # Filter for XYZ tile sources, excluding static map APIs
    valid_sources = {
        k: v
        for k, v in BasemapConfig.BASEMAP_TILES.items()
        if "{z}" in v and "{x}" in v and "{y}" in v
    }

    if source not in valid_sources:
        raise ValueError(
            f"Invalid XYZ tile source: {source}. Valid sources are: {list(valid_sources.keys())}"
        )

    url_template = valid_sources[source]
    xtile, ytile = deg2num(lat, lon, zoom)
    url = url_template.format(z=zoom, x=xtile, y=ytile)

    response = requests.get(url)
    response.raise_for_status()
    return response.content
