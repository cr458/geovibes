import pytest
import shapely.geometry
import pyproj
import tempfile
import geopandas as gpd
import os

from geovibes.tiling import MGRSTileId, MGRSTileGrid, get_crs_from_mgrs_tile_id, get_mgrs_tile_ids_for_roi


# Test data for MGRSTileId
@pytest.mark.parametrize("mgrs_id,expected_utm_zone,expected_latitude_band,expected_grid_square", [
    ("18SJH", 18, "S", "JH"),
    ("32NEA", 32, "N", "EA"),
    ("1AAB", 1, "A", "AB"),
])
def test_mgrs_tile_id_from_str_valid(mgrs_id, expected_utm_zone, expected_latitude_band, expected_grid_square):
    """Test MGRSTileId.from_str with valid MGRS IDs."""
    tile_id = MGRSTileId.from_str(mgrs_id)
    assert tile_id.utm_zone == expected_utm_zone
    assert tile_id.latitude_band == expected_latitude_band
    assert tile_id.grid_square == expected_grid_square
    assert str(tile_id) == mgrs_id


@pytest.mark.parametrize("invalid_mgrs_id", [
    "123",  # Too short
    "123456",  # Too long
])
def test_mgrs_tile_id_from_str_invalid(invalid_mgrs_id):
    """Test MGRSTileId.from_str with invalid MGRS IDs."""
    with pytest.raises(ValueError):
        MGRSTileId.from_str(invalid_mgrs_id)


@pytest.mark.parametrize("utm_zone,latitude_band,grid_square,expected_str", [
    (18, "S", "JH", "18SJH"),
    (32, "N", "EA", "32NEA"),
    (1, "A", "AB", "1AAB"),
])
def test_mgrs_tile_id_creation_and_str(utm_zone, latitude_band, grid_square, expected_str):
    """Test MGRSTileId creation and string representation."""
    tile_id = MGRSTileId(utm_zone=utm_zone, latitude_band=latitude_band, grid_square=grid_square)
    assert str(tile_id) == expected_str
    assert tile_id.utm_zone == utm_zone
    assert tile_id.latitude_band == latitude_band
    assert tile_id.grid_square == grid_square


@pytest.mark.parametrize("utm_zone,latitude_band,grid_square", [
    (0, "S", "JH"),  # UTM zone too low
    (61, "S", "JH"),  # UTM zone too high
    (18, "SS", "JH"),  # Latitude band too long
    (18, "", "JH"),  # Latitude band empty
    (18, "S", "J"),  # Grid square too short
    (18, "S", "JHH"),  # Grid square too long
])
def test_mgrs_tile_id_invalid_creation(utm_zone, latitude_band, grid_square):
    """Test MGRSTileId creation with invalid parameters."""
    with pytest.raises(ValueError):
        MGRSTileId(utm_zone=utm_zone, latitude_band=latitude_band, grid_square=grid_square)


def test_mgrs_tile_id_roundtrip():
    """Test that creating from string and converting back gives the same result."""
    original_id = "18SJH"
    tile_id = MGRSTileId.from_str(original_id)
    assert str(tile_id) == original_id


@pytest.mark.parametrize("mgrs_id,expected_epsg", [
    ("25SEA", 32625),
    ("19KET", 32719),  
])
def test_mgrs_tile_id_crs_property(mgrs_id, expected_epsg):
    """Test MGRSTileId.crs property returns correct EPSG code."""
    tile_id = MGRSTileId.from_str(mgrs_id)
    crs = get_crs_from_mgrs_tile_id(tile_id)
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == expected_epsg

# Test data for get_mgrs_tile_ids_for_roi
@pytest.fixture
def sample_mgrs_tiles_file():
    """Create a temporary file with sample MGRS tiles for testing."""
    # Create sample MGRS tiles data
    tiles_data = {
        'mgrs_id': ['18SJH', '18SJI', '18SJJ', '32NEA', '32NEB'],
        'geometry': [
            shapely.geometry.box(-80, 30, -79, 31),  # 18SJH
            shapely.geometry.box(-79, 30, -78, 31),  # 18SJI
            shapely.geometry.box(-78, 30, -77, 31),  # 18SJJ
            shapely.geometry.box(10, 50, 11, 51),    # 32NEA
            shapely.geometry.box(11, 50, 12, 51),    # 32NEB
        ]
    }
    
    gdf = gpd.GeoDataFrame(tiles_data, crs="EPSG:4326")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        gdf.to_parquet(tmp_file.name)
        yield tmp_file.name
    
    # Cleanup
    os.unlink(tmp_file.name)


def test_get_mgrs_tile_ids_for_roi_basic(sample_mgrs_tiles_file):
    """Test basic functionality of get_mgrs_tile_ids_for_roi."""
    # Create a search geometry that intersects with 18SJH and 18SJI
    search_geometry = shapely.geometry.box(-79.5, 30.5, -78.5, 30.5)
    
    result = get_mgrs_tile_ids_for_roi(
        search_geometry=search_geometry,
        search_geometry_crs="EPSG:4326",
        mgrs_tiles_file=sample_mgrs_tiles_file
    )
    
    assert len(result) == 2
    mgrs_ids = [str(tile_id) for tile_id in result]
    assert "18SJH" in mgrs_ids
    assert "18SJI" in mgrs_ids


def test_get_mgrs_tile_ids_for_roi_no_intersection(sample_mgrs_tiles_file):
    """Test get_mgrs_tile_ids_for_roi with geometry that doesn't intersect any tiles."""
    # Create a search geometry that doesn't intersect any tiles
    search_geometry = shapely.geometry.box(-90, 0, -89, 1)
    
    result = get_mgrs_tile_ids_for_roi(
        search_geometry=search_geometry,
        search_geometry_crs="EPSG:4326",
        mgrs_tiles_file=sample_mgrs_tiles_file
    )
    
    assert len(result) == 0


def test_get_mgrs_tile_ids_for_roi_all_tiles(sample_mgrs_tiles_file):
    """Test get_mgrs_tile_ids_for_roi with geometry that covers all tiles."""
    # Create a search geometry that covers all tiles
    search_geometry = shapely.geometry.box(-90, 0, 20, 60)
    
    result = get_mgrs_tile_ids_for_roi(
        search_geometry=search_geometry,
        search_geometry_crs="EPSG:4326",
        mgrs_tiles_file=sample_mgrs_tiles_file
    )
    
    assert len(result) == 5
    mgrs_ids = [str(tile_id) for tile_id in result]
    expected_ids = ['18SJH', '18SJI', '18SJJ', '32NEA', '32NEB']
    for expected_id in expected_ids:
        assert expected_id in mgrs_ids


@pytest.mark.parametrize(
    "mgrs_id_str, expected_epsg, tilesize, overlap, resolution, expected_prefix",
    [
        (
            "13SFA",
            32613,
            25,
            0,
            10.0,
            "13SFA_32613_25_0_10",
        ),  # Northern Hemisphere ('S' band is northern)
        (
            "13LFA",
            32713,
            25,
            0,
            10.0,
            "13LFA_32713_25_0_10",
        ),  # Southern Hemisphere
        (
            "18TXL",
            32618,
            50,
            5,
            2.5,
            "18TXL_32618_50_5_2",
        ),  # Northern Hemisphere
    ],
)
def test_mgrs_grid_and_crs_properties(
    mgrs_id_str, expected_epsg, tilesize, overlap, resolution, expected_prefix
):
    """
    Tests the crs property on MGRSTileId and ensures MGRSTileGrid
    is instantiated correctly. This test covers regressions for issues
    found in tiling_to_gee_asset.py.
    """
    mgrs_tile_id = MGRSTileId.from_str(mgrs_id_str)

    # Test MGRSTileId.crs
    assert isinstance(mgrs_tile_id.crs, pyproj.CRS)
    assert mgrs_tile_id.crs.to_epsg() == expected_epsg

    # Test MGRSTileGrid initialization and prefix
    grid = MGRSTileGrid(
        mgrs_tile_id=mgrs_tile_id,
        tilesize=tilesize,
        overlap=overlap,
        resolution=resolution,
    )
    assert grid.prefix == expected_prefix