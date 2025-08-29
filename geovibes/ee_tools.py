from typing import Dict, Optional
from dotenv import load_dotenv
import ee


def initialize_ee_with_credentials(
    project: Optional[str] = None, verbose: bool = False
) -> bool:
    """Initialize Earth Engine with user credentials.

    Args:
        project: Google Cloud project ID. If None, uses EE default.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    load_dotenv()
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        if verbose:
            print("✅ Earth Engine initialized with user credentials")
        return True
    except Exception as e:
        if verbose:
            print(f"❌ Earth Engine authentication failed: {e}")
            print(
                "\n🔧 To enable NDVI/NDWI basemaps, please run the following command:"
            )
            print("    earthengine authenticate")
            print(
                "⚠️  Continuing without Earth Engine (NDVI/NDWI basemaps will be unavailable)"
            )
        return False


def get_s2_cloud_masked_collection(
    aoi: Optional[ee.Geometry],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    clear_threshold: float = 0.80,
) -> ee.ImageCollection:
    """Get cloud-masked Sentinel-2 collection using CloudScore+ for quality filtering.

    Args:
        aoi: Earth Engine geometry defining area of interest. If None, no spatial filter is applied.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        clear_threshold: CloudScore+ threshold (0-1) where 1 is clearest.

    Returns:
        Sentinel-2 image collection with cloud masking applied.
    """
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    QA_BAND = "cs_cdf"

    collection = s2.filterDate(start_date, end_date)
    if aoi:
        collection = collection.filterBounds(aoi)

    return (
        collection.linkCollection(csPlus, [QA_BAND])
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
    )


def get_s2_rgb_median(
    aoi: Optional[ee.Geometry],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    clear_threshold: float = 0.80,
) -> ee.Image:
    """Create median RGB composite from cloud-masked Sentinel-2 imagery.

    Args:
        aoi: Earth Engine geometry defining area of interest. If None, no spatial filter is applied.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        clear_threshold: CloudScore+ threshold (0-1) for pixel quality.

    Returns:
        Median RGB composite with bands B4 (red), B3 (green), B2 (blue).
    """
    collection = get_s2_cloud_masked_collection(
        aoi, start_date, end_date, clear_threshold
    )
    return collection.select(["B4", "B3", "B2"]).median()


def get_s2_ndvi_median(
    aoi: Optional[ee.Geometry],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    clear_threshold: float = 0.80,
) -> ee.Image:
    """Create median NDVI from cloud-masked Sentinel-2 imagery.

    Args:
        aoi: Earth Engine geometry defining area of interest. If None, no spatial filter is applied.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        clear_threshold: CloudScore+ threshold (0-1) for pixel quality.

    Returns:
        Median NDVI image calculated from NIR (B8) and red (B4) bands.
    """
    collection = get_s2_cloud_masked_collection(
        aoi, start_date, end_date, clear_threshold
    )
    ndvi = collection.map(
        lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    )
    return ndvi.median()


def get_s2_ndwi_median(
    aoi: Optional[ee.Geometry],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    clear_threshold: float = 0.80,
) -> ee.Image:
    """Create median NDWI from cloud-masked Sentinel-2 imagery.

    Args:
        aoi: Earth Engine geometry defining area of interest. If None, no spatial filter is applied.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        clear_threshold: CloudScore+ threshold (0-1) for pixel quality.

    Returns:
        Median NDWI image calculated from green (B3) and NIR (B8) bands.
    """
    collection = get_s2_cloud_masked_collection(
        aoi, start_date, end_date, clear_threshold
    )
    ndwi = collection.map(
        lambda img: img.normalizedDifference(["B3", "B8"]).rename("ndwi")
    )
    return ndwi.median()


def get_s2_hsv_median(
    aoi: Optional[ee.Geometry],
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    clear_threshold: float = 0.80,
) -> ee.Image:
    """Create median HSV composite from cloud-masked Sentinel-2 imagery.

    Args:
        aoi: Earth Engine geometry defining area of interest. If None, no spatial filter is applied.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        clear_threshold: CloudScore+ threshold (0-1) for pixel quality.

    Returns:
        HSV composite with hue, saturation, and value bands.
    """
    rgb_median = get_s2_rgb_median(aoi, start_date, end_date, clear_threshold).divide(
        10000
    )
    hsv_median = ee.Image.rgbToHsv(rgb_median)
    return hsv_median.select(["hue", "saturation", "value"])


def get_ee_image_url(image: ee.Image, vis_params: Dict) -> str:
    """Generate tile URL template for Earth Engine image display.

    Args:
        image: Earth Engine image to visualize.
        vis_params: Visualization parameters (min, max, palette, bands, etc.).

    Returns:
        URL template string with {z}/{x}/{y} placeholders for map tiles.
    """
    map_id = image.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format
