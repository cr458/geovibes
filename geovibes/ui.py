"""Interactive map interface for geospatial similarity search using satellite embeddings."""

import base64
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional

import duckdb
import ee
import geopandas as gpd
import ipyleaflet as ipyl
import ipywidgets as ipyw
from ipyleaflet import Map, DrawControl
from IPython.display import display
from ipywidgets import (
    Button,
    VBox,
    HBox,
    IntSlider,
    Label,
    Layout,
    HTML,
    ToggleButtons,
    Accordion,
    FileUpload,
    Dropdown,
)
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
import webbrowser
from PIL import Image as PILImage, ImageDraw
import base64
from io import BytesIO
import faiss
import pathlib

# --- Crash Debugging Logger ---
LOG_FILE = "geovibes_crash.log"
def _log_to_file(message):
    """Appends a timestamped message to the log file immediately."""
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\\n")
# -----------------------------

from .ee_tools import (
    get_s2_rgb_median,
    get_s2_ndvi_median,
    get_s2_ndwi_median,
    get_s2_hsv_median,
    get_ee_image_url,
)
from .ui_config import (
    UIConstants,
    BasemapConfig,
    GeoVibesConfig,
    DatabaseConstants,
    LayerStyles,
)
from .ee_tools import initialize_ee_with_credentials
from .utils import list_databases_in_directory, get_database_centroid
from .xyz import get_map_image

warnings.simplefilter("ignore", category=FutureWarning)

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn(
        "MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key."
    )


class GeoVibes:
    """Interactive map interface for geospatial similarity search using satellite embeddings.

    Provides point-and-click labeling interface with similarity search capabilities
    using vector embeddings and Faiss indexing.
    """

    @classmethod
    def from_config(cls, config_path, verbose=False, **kwargs):
        """Create a GeoVibes instance from a configuration file (deprecated).

        Args:
            config_path: Path to JSON configuration file
            verbose: If True, print detailed progress messages
            **kwargs: Additional keyword arguments to override config values

        Returns:
            GeoVibes instance
        """
        if verbose:
            print(
                "⚠️  from_config() is deprecated. Use GeoVibes() with individual parameters instead."
            )
        return cls(config_path=config_path, verbose=verbose, **kwargs)

    @classmethod
    def create(cls, 
              duckdb_path: Optional[str] = None,
              duckdb_directory: Optional[str] = None,
              boundary_path: Optional[str] = None,
              start_date: str = "2024-01-01",
              end_date: str = "2025-01-01",
              gcp_project: Optional[str] = None,
              verbose: bool = False,
              **kwargs):
        """Create a GeoVibes instance with explicit parameters.

        Args:
            duckdb_path: Path to DuckDB database file
            duckdb_directory: Directory containing multiple DuckDB database files
            boundary_path: Path to boundary GeoJSON file
            start_date: Start date in YYYY-MM-DD format for Earth Engine basemaps
            end_date: End date in YYYY-MM-DD format for Earth Engine basemaps
            gcp_project: Google Cloud Project ID for Earth Engine authentication
            verbose: Enable detailed progress messages
            **kwargs: Additional arguments

        Returns:
            GeoVibes instance
        """
        return cls(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            gcp_project=gcp_project,
            verbose=verbose,
            **kwargs,
        )

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
            baselayer_url: Optional[str] = None,
            disable_ee: bool = False,
            verbose: bool = False, 
            **kwargs) -> None:
        """Initialize GeoVibes interface.

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
            baselayer_url: Custom basemap tile URL.
            disable_ee: Disable Earth Engine basemaps.
            verbose: Enable detailed progress messages.
            **kwargs: Additional arguments for backwards compatibility.

        Raises:
            ValueError: If no database is available given the provided parameters.
            FileNotFoundError: If no .db files are found in the provided directory.
            RuntimeError: If there is an error connecting to the database.
        """
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes...")

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
        
        self.ee_available = not disable_ee and initialize_ee_with_credentials(self.config.gcp_project)
        self.faiss_index = None
        
        # Initialize database list if directory is provided
        self.available_databases = []
        self.current_database_path = None
        self.current_faiss_path = None
        if self.config.duckdb_directory:
            self.available_databases = list_databases_in_directory(
                self.config.duckdb_directory, verbose=self.verbose
            )
            if self.available_databases:
                db_info = self.available_databases[0]
                self.current_database_path = db_info["db_path"]
                self.current_faiss_path = db_info["faiss_path"]
                if self.verbose:
                    print(
                        f"📁 Found {len(self.available_databases)} databases in directory"
                    )
            else:
                raise FileNotFoundError("⚠️  No .db files found in directory")

        if baselayer_url is None:
            baselayer_url = BasemapConfig.BASEMAP_TILES["MAPTILER"]

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
            
            # Configure memory limits and disable progress bar to prevent kernel crashes
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)
            
            # Extra insurance: explicitly disable progress bar again
            try:
                self.duckdb_connection.execute("SET enable_progress_bar=false")
                self.duckdb_connection.execute("SET enable_profiling=false")
                self.duckdb_connection.execute("SET enable_object_cache=false")
                if self.verbose:
                    print("✅ Progress bar and profiling disabled")
            except:
                pass  # Ignore if these settings don't exist
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False
        self.current_basemap = "MAPTILER"
        self.basemap_layer = ipyl.TileLayer(
            url=baselayer_url,
            no_wrap=True,
            name="basemap",
            attribution="",  # Empty attribution since we're using custom control
        )
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
            extension_queries = DatabaseConstants.get_extension_setup_queries(self.current_database_path)
            for query in extension_queries:
                try:
                    self.duckdb_connection.execute(query)
                    if self.verbose:
                        if "httpfs" in query:
                            print("📦 httpfs extension loaded for GCS support")
                        elif "spatial" in query:
                            print("🗺️  spatial extension loaded for geometry support")
                except Exception as e:
                    raise RuntimeError(f"Failed to load required extension: {str(e)}")

        # Load FAISS index if specified
        if not self.current_faiss_path:
            raise ValueError("Could not find a FAISS index for the selected database.")
        if self.verbose:
            print(f"🧠 Loading FAISS index from: {self.current_faiss_path}")
        self.faiss_index = faiss.read_index(self.current_faiss_path)
        if self.verbose:
            print(f"✅ FAISS index loaded. Contains {self.faiss_index.ntotal} vectors.")

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
        center_y, center_x = self._setup_boundary_and_center()

        # Build map
        self.map = self._build_map(center_y, center_x)

        # Add Earth Engine basemap options (if available)
        self._setup_ee_basemaps()

        if self.verbose:
            print("Building UI...")

        # Initialize state
        self.current_label = "Positive"
        self.execute_label_point = True
        self.select_val = UIConstants.POSITIVE_LABEL  # Initialize to positive
        self.pos_ids = []
        self.neg_ids = []
        self.detection_gdf = None
        self.lasso_mode = False
        self.query_vector = None
        self.detection_ids = []
        self.cached_embeddings = {}
        self.detections_with_embeddings = None
        self.current_operation = None  # Track current operation for status display
        self.vector_layer = None  # Track custom vector layer

        # New state for tiles panel
        self.tile_basemap = self.current_basemap
        self.tile_page = 0
        self.tiles_per_page = 50 # Subsequent page size
        self.initial_load_size = 8
        self.last_search_results_df = None

        # Build UI
        self.side_panel, self.ui_widgets = self._build_side_panel()

        # Build tiles panel
        self._build_tiles_panel()

        # Add layers to map
        self._add_map_layers()

        # Update boundary layer if we have one
        self._update_boundary_layer()

        # Add DrawControl
        self._setup_draw_control()

        # Wire events
        self._wire_events()

        # Add legend
        self.legend = HTML(
            value=f"""
            <div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>
                <div><strong>Labels:</strong> 
                    <span style='color: {UIConstants.POS_COLOR}; font-weight: bold;'>🔵 Positive</span> | 
                    <span style='color: {UIConstants.NEG_COLOR}; font-weight: bold;'>🟠 Negative</span>
                </div>
                <div style='margin-top: 3px;'><strong>Search Results:</strong> 
                    <span style='color: #00ff00; font-weight: bold;'>🟢 Most Similar</span> → 
                    <span style='color: #ffff00; font-weight: bold;'>🟡 Medium</span> → 
                    <span style='color: #ff4444; font-weight: bold;'>🔴 Least Similar</span>
                </div>
            </div>
        """
        )

        # Add status bar
        self.status_bar = HTML(value="Ready")



        # Create main layout
        map_with_overlays = VBox(
            [
                self.map,
                HBox(
                    [self.legend, self.status_bar],
                    layout=Layout(justify_content="space-between", padding="5px"),
                ),
            ],
            layout=Layout(flex="1 1 auto"),
        )

        self.main_layout = HBox(
            [self.side_panel, map_with_overlays],
            layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width="100%"),
        )

        display(self.main_layout)

    def _setup_ee_basemaps(self) -> None:
        """Set up Earth Engine basemaps (Sentinel-2 RGB, NDVI, NDWI) if available."""
        self.basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()

        if self.ee_available:
            try:
                if self.verbose:
                    print("🛰️ Setting up Earth Engine basemaps (S2 RGB, NDVI, NDWI, HSV)...")

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
                self.basemap_tiles["S2_NDVI"] = ndvi_url

                ndwi_median = get_s2_ndwi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date
                )
                ndwi_url = get_ee_image_url(ndwi_median, BasemapConfig.NDWI_VIS_PARAMS)
                self.basemap_tiles["S2_NDWI"] = ndwi_url

                hsv_median = get_s2_hsv_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date
                )
                hsv_url = get_ee_image_url(hsv_median, BasemapConfig.S2_HSV_VIS_PARAMS)
                self.basemap_tiles["S2_HSV"] = hsv_url

                if self.verbose:
                    print("✅ Earth Engine basemaps added successfully!")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to create Earth Engine basemaps: {e}")
                    print("⚠️  Continuing with basic basemaps only")
        else:
            if not self.ee_available and self.verbose:
                print("⚠️  Earth Engine not available - S2/NDVI/NDWI basemaps skipped")

    def _build_map(self, center_y, center_x):
        """Build and return the map widget."""
        map_widget = Map(
            basemap=self.basemap_layer,
            center=(center_y, center_x),
            zoom=UIConstants.DEFAULT_ZOOM,
            layout=Layout(flex="1 1 auto", height="100%"),
            scroll_wheel_zoom=True,
            attribution_control=False,  # Disable ALL attribution controls
        )
        
        # Add custom attribution control positioned at bottom left with proper attribution text
        attribution_control = ipyl.AttributionControl(
            position='bottomleft',
            prefix='<a href="https://leafletjs.com">Leaflet</a> | ' + BasemapConfig.MAPTILER_ATTRIBUTION
        )
        map_widget.add_control(attribution_control)
        
        return map_widget

    def _build_side_panel(self):
        """Build the collapsible side panel with accordion sections."""
        self.search_btn = Button(
            description="Search",
            layout=Layout(flex="1", height=UIConstants.BUTTON_HEIGHT),
            button_style="success",  # Green to highlight importance
            tooltip="Find points similar to your positive labels",
        )

        self.tiles_button = Button(
            description="",
            icon="th",
            layout=Layout(width="40px", height=UIConstants.BUTTON_HEIGHT),
            button_style="",
            tooltip="View search results as tiles",
        )

        search_controls = HBox([self.search_btn, self.tiles_button])

        self.neighbors_slider = IntSlider(
            value=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            description="",  # No description
            readout=True,
            layout=Layout(width="100%"),
        )

        self.reset_btn = Button(
            description="🗑️ Reset",
            layout=Layout(width="100%", height=UIConstants.RESET_BUTTON_HEIGHT),
            button_style="",
            tooltip="Clear all labels and search results",
        )

        search_section = VBox(
            [search_controls, self.neighbors_slider, self.reset_btn],
            layout=Layout(padding="5px", margin="0 0 10px 0"),
        )

        # --- Labeling section ---
        self.label_toggle = ToggleButtons(
            options=[
                ("Positive", "Positive"),
                ("Negative", "Negative"),
                ("Erase", "Erase"),
            ],
            value="Positive",
            layout=Layout(width="100%"),
        )

        # Add selection mode toggle
        self.selection_mode = ToggleButtons(
            options=[("Point", "point"), ("Polygon", "polygon")],
            value="point",
            layout=Layout(width="100%"),
        )

        # Apply colors to toggle buttons
        self._update_toggle_button_styles()

        # --- Basemap Selection ---
        self.basemap_buttons = {}
        basemap_section_widgets = []

        # Use instance basemap_tiles which includes EE basemaps (S2_RGB, S2_NDVI, S2_NDWI, S2_HSV)
        basemap_tiles_to_use = self.basemap_tiles

        for basemap_name in basemap_tiles_to_use.keys():
            btn = Button(
                description=basemap_name.replace("_", " "),
                layout=Layout(width="100%", margin="1px"),
                button_style="",
            )
            btn.basemap_name = basemap_name  # Store basemap name for reference
            self.basemap_buttons[basemap_name] = btn
            basemap_section_widgets.append(btn)

        # Highlight current basemap
        self._update_basemap_button_styles()

        # --- Export section ---
        self.save_btn = Button(
            description="💾 Save Dataset", layout=Layout(width="100%")
        )

        # --- Load Dataset section ---
        self.load_btn = Button(
            description="📂 Load Dataset", layout=Layout(width="100%")
        )
        self.file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none"),  # Initially hidden
        )

        # --- Database Selection section ---
        self.database_dropdown = None
        database_section_widgets = []
        if self.available_databases:
            # Create dropdown with database names (showing just filenames for clarity)
            database_options = [
                (os.path.basename(db["db_path"]), db["db_path"])
                for db in self.available_databases
            ]
            self.database_dropdown = Dropdown(
                options=database_options,
                value=self.current_database_path,
                description="",
                layout=Layout(width="100%"),
            )
            database_section_widgets.append(Label("Select Database:"))
            database_section_widgets.append(self.database_dropdown)

        # --- Add Vector Layer section ---
        self.add_vector_btn = Button(
            description="📄 Add Vector Layer",
            layout=Layout(width="100%"),
            button_style="",
        )
        self.vector_file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none"),  # Initially hidden
        )

        # --- External Tools section ---
        self.google_maps_btn = Button(
            description="🌍 Google Maps ↗", layout=Layout(width="100%"), button_style=""
        )

        # --- Run Button ---
        self.run_button = Button(description=f"Find Similar", button_style='primary', layout=Layout(width='120px'))

        # Build accordion - conditionally include database section
        accordion_children = [
            VBox(
                [
                    Label("Label Type:"),
                    self.label_toggle,
                    Label("Selection Mode:", layout=Layout(margin="10px 0 0 0")),
                    self.selection_mode,
                ],
                layout=Layout(padding="5px"),
            ),
            VBox(basemap_section_widgets, layout=Layout(padding="5px")),
            VBox(
                [
                    self.save_btn,
                    self.load_btn,
                    self.file_upload,
                    self.add_vector_btn,
                    self.vector_file_upload,
                    self.google_maps_btn,
                    self.run_button,
                ],
                layout=Layout(padding="5px"),
            ),
        ]

        accordion_titles = ["Label Mode", "Basemaps", "Export & Tools"]

        # Add database section if available
        if database_section_widgets:
            accordion_children.insert(
                0, VBox(database_section_widgets, layout=Layout(padding="5px"))
            )
            accordion_titles.insert(0, "Database")

        accordion = Accordion(children=accordion_children)

        # Set titles
        for i, title in enumerate(accordion_titles):
            accordion.set_title(i, title)

        # Open label mode by default
        accordion.selected_index = 0

        # Add collapse/expand functionality
        self.panel_collapsed = False
        self.collapse_btn = Button(
            description="◀",
            layout=Layout(
                width=UIConstants.COLLAPSE_BUTTON_SIZE,
                height=UIConstants.COLLAPSE_BUTTON_SIZE,
            ),
            tooltip="Collapse/Expand Panel",
        )

        # Main panel with collapse button
        panel_header = HBox(
            [Label("Controls", layout=Layout(flex="1")), self.collapse_btn],
            layout=Layout(width="100%", justify_content="space-between", padding="2px"),
        )

        # Create accordion container that will be hidden/shown
        self.accordion_container = VBox([accordion], layout=Layout(width="100%"))

        # Panel content includes search (always visible) and accordion (collapsible)
        panel_content = VBox(
            [
                panel_header,
                search_section,  # Always visible
                self.accordion_container,  # This will be hidden/shown
            ],
            layout=Layout(width=UIConstants.PANEL_WIDTH, padding="5px"),
        )  # Narrower width

        # Return panel and widget references
        ui_widgets = {
            "search_btn": self.search_btn,
            "reset_btn": self.reset_btn,
            "label_toggle": self.label_toggle,
            "selection_mode": self.selection_mode,
            "neighbors_slider": self.neighbors_slider,
            "basemap_buttons": self.basemap_buttons,
            "save_btn": self.save_btn,
            "load_btn": self.load_btn,
            "file_upload": self.file_upload,
            "add_vector_btn": self.add_vector_btn,
            "vector_file_upload": self.vector_file_upload,
            "google_maps_btn": self.google_maps_btn,
            "collapse_btn": self.collapse_btn,
            "tiles_button": self.tiles_button,
            "run_button": self.run_button,
        }

        return panel_content, ui_widgets

    def _create_placeholder_png(self, lat, lon, size=(256, 256)):
        """Create a placeholder PNG image for a given lat/lon coordinate.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            size: Tuple of (width, height) for the image

        Returns:
            Base64 encoded PNG image data
        """
        # Create a simple placeholder image with gradient and coordinates
        img = PILImage.new("RGB", size, color="lightblue")
        draw = ImageDraw.Draw(img)

        # Add a simple gradient effect
        for y in range(size[1]):
            color_intensity = int(255 * (1 - y / size[1]))
            draw.line(
                [(0, y), (size[0], y)], fill=(color_intensity, color_intensity, 255)
            )

        # Add coordinate text (truncated to fit)
        coord_text = f"{lat:.2f},{lon:.2f}"

        # Create a simple border
        draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline="darkblue", width=2)

        # Add a small circle in the center to represent the point
        center_x, center_y = size[0] // 2, size[1] // 2
        draw.ellipse(
            [center_x - 3, center_y - 3, center_x + 3, center_y + 3],
            fill="red",
            outline="darkred",
        )

        # Convert to base64 for embedding in HTML
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_data}"


    def _build_tiles_panel(self):
        """Build the results panel with controls."""
        basemap_options = list(BasemapConfig.BASEMAP_TILES.keys())
        self.tile_basemap_dropdown = ipyw.Dropdown(
            options=basemap_options,
            value=self.tile_basemap,
            description="",
            layout=ipyw.Layout(width="180px"),
            style={'description_width': 'initial'}
        )

        self.next_tiles_btn = Button(
            description="Next", 
            layout=ipyw.Layout(width="60px", margin="0 0 0 5px", display='none')
        )
        
        tiles_controls = ipyw.HBox(
            [self.tile_basemap_dropdown, self.next_tiles_btn],
            layout=ipyw.Layout(
                justify_content="flex-start",  # Align items to the left
                align_items="center",
                width="100%",
                margin="0 0 10px 0"
            ),
        )

        self.tiles_display = ipyw.Output()
        self.results_grid = ipyw.GridBox(
            [],
            layout=Layout(
                width="100%",
                grid_template_columns="1fr 1fr",
                grid_gap="3px",
            ),
        )
        with self.tiles_display:
            display(self.results_grid)

        # Create a scrollable container for just the tiles
        tiles_scroll_container = VBox(
            [self.tiles_display],
            layout=Layout(
                width="100%",
                max_height="600px",
                overflow_y="auto",
            ),
        )

        self.tiles_pane = VBox(
            [tiles_controls, tiles_scroll_container],
            layout=Layout(
                display="none",
                width="265px",  # Tighter width for compact layout
                padding="5px",
            ),
        )
        tiles_pane_control = ipyl.WidgetControl(widget=self.tiles_pane, position="topright")
        self.map.add_control(tiles_pane_control)

        # Wire events
        self.tile_basemap_dropdown.observe(self._on_tile_basemap_change, names="value")
        self.next_tiles_btn.on_click(self._on_next_tiles_click)

    def _on_tiles_click(self, b):
        """Toggle the tiles panel visibility."""
        if self.tiles_button.button_style == 'success':
            if self.tiles_pane.layout.display == 'none':
                self.tiles_pane.layout.display = ''
            else:
                self.tiles_pane.layout.display = 'none'
    
    def _on_tile_click(self, button):
        """Handle tile click for labeling."""
        point_id = button.point_id
        row_data = button.row_data
        
        # Fetch embedding if not cached
        if point_id not in self.cached_embeddings:
            self._fetch_embeddings([point_id])
        
        # Remove from existing labels
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)
        
        # Add to appropriate label list based on current label mode
        if self.select_val == UIConstants.POSITIVE_LABEL:
            self.pos_ids.append(point_id)
            new_border_color = UIConstants.POS_COLOR
            new_border_width = "3px"
            if self.verbose:
                print(f"✅ Labeled point {point_id} as Positive")
        elif self.select_val == UIConstants.NEGATIVE_LABEL:
            self.neg_ids.append(point_id)
            new_border_color = UIConstants.NEG_COLOR
            new_border_width = "3px"
            if self.verbose:
                print(f"✅ Labeled point {point_id} as Negative")
        else:  # Erase mode
            new_border_color = "#ccc"
            new_border_width = "1px"
            if self.verbose:
                print(f"✅ Erased label for point {point_id}")
        
        # Update the button border immediately
        button.layout.border = f"{new_border_width} solid {new_border_color}"
        
        # Update visualization layers and query vector
        self.update_layers()
        self.update_query_vector()
        
        # Show status
        if self.select_val != UIConstants.ERASE_LABEL:
            self._show_operation_status(f"✅ Labeled tile as {self.current_label}")
        else:
            self._show_operation_status(f"✅ Erased label from tile")
    
    def _on_tile_label_click(self, button):
        """Handle tick/cross button click for labeling tiles."""
        point_id = button.point_id
        row_data = button.row_data
        is_positive = button.is_positive
        partner_button = button.partner_button
        
        # Fetch embedding if not cached
        if point_id not in self.cached_embeddings:
            self._fetch_embeddings([point_id])
        
        # Check if this label is already selected
        was_selected = (is_positive and point_id in self.pos_ids) or (not is_positive and point_id in self.neg_ids)
        
        # Remove from all existing labels
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)
        
        # If it wasn't selected, add the new label
        if not was_selected:
            if is_positive:
                self.pos_ids.append(point_id)
                # Update button appearances
                button.button_style = "primary"
                button.layout.opacity = "1.0"
                partner_button.button_style = ""
                partner_button.layout.opacity = "0.3"
                if self.verbose:
                    print(f"✅ Labeled point {point_id} as Positive")
                self._show_operation_status(f"✅ Labeled tile as Positive")
            else:
                self.neg_ids.append(point_id)
                # Update button appearances
                button.button_style = "warning"
                button.layout.opacity = "1.0"
                partner_button.button_style = ""
                partner_button.layout.opacity = "0.3"
                if self.verbose:
                    print(f"✅ Labeled point {point_id} as Negative")
                self._show_operation_status(f"✅ Labeled tile as Negative")
        else:
            # If it was selected, we're removing the label (toggle off)
            button.button_style = ""
            button.layout.opacity = "0.3"
            if self.verbose:
                print(f"✅ Removed label from point {point_id}")
            self._show_operation_status(f"✅ Removed label from tile")
        
        # Update visualization layers and query vector
        self.update_layers()
        self.update_query_vector()
    
    def _on_tile_map_click(self, button):
        """Handle map button click to pan and zoom to tile location."""
        point_id = button.point_id
        row_data = button.row_data
        
        # Pan and zoom to the tile location
        try:
            geom = shapely.wkt.loads(row_data["geometry_wkt"])
            lat, lon = geom.y, geom.x
            self.map.center = (lat, lon)
            self.map.zoom = 14  # Closer zoom for better detail
            
            # Create a small square polygon around the point
            half_size = 0.0025 / 2  # Half of 0.0025 degrees
            square_coords = [
                (lon - half_size, lat - half_size),
                (lon + half_size, lat - half_size),
                (lon + half_size, lat + half_size),
                (lon - half_size, lat + half_size),
                (lon - half_size, lat - half_size)  # Close the polygon
            ]
            
            # Create the polygon and add it to the map
            from shapely.geometry import Polygon
            square_poly = Polygon(square_coords)
            
            # Remove any existing highlight layer
            for layer in self.map.layers:
                if hasattr(layer, 'name') and layer.name == 'tile_highlight':
                    self.map.remove_layer(layer)
            
            # Add the highlight square
            highlight_layer = ipyl.GeoJSON(
                data={
                    "type": "Feature",
                    "geometry": shapely.geometry.mapping(square_poly),
                    "properties": {"id": point_id}
                },
                style={
                    'color': '#ff0000',
                    'weight': 3,
                    'fillOpacity': 0
                },
                name='tile_highlight'
            )
            self.map.add_layer(highlight_layer)
            
            self._show_operation_status(f"📍 Centered on tile {point_id}")
            if self.verbose:
                print(f"📍 Panned to tile {point_id} at ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            if self.verbose:
                print(f"Could not pan to tile location: {e}")
            self._show_operation_status("⚠️ Could not pan to tile location")
    


    def _on_tile_basemap_change(self, change):
        """Handle tile basemap change."""
        self.tile_basemap = change["new"]
        
        # Convert all existing tiles to loading placeholders
        current_tile_count = len(self.results_grid.children)
        if current_tile_count > 0:
            loading_tiles = []
            for i in range(current_tile_count):
                loading_label = ipyw.Label(
                    value="Loading...",
                    layout=ipyw.Layout(
                        width="100px", 
                        height="100px", 
                        border="1px solid #ccc",
                        display="flex",
                        align_items="center",
                        justify_content="center"
                    )
                )
                loading_tiles.append(loading_label)
            self.results_grid.children = loading_tiles
            
            # Reload all tiles with new basemap
            self._reload_all_tiles_with_new_basemap()

    def _reload_all_tiles_with_new_basemap(self):
        """Reload all currently displayed tiles with the new basemap."""
        if self.last_search_results_df is None or self.last_search_results_df.empty:
            return
            
        # Calculate how many tiles are currently displayed
        current_tile_count = len(self.results_grid.children)
        total_pages_displayed = (current_tile_count + self.tiles_per_page - 1) // self.tiles_per_page
        
        # Get all the data for currently displayed tiles
        end_index = total_pages_displayed * self.tiles_per_page
        all_displayed_df = self.last_search_results_df.iloc[:end_index]
        
        def create_and_update_tile(idx, row):
            try:
                geom = shapely.wkt.loads(row["geometry_wkt"])
                image_bytes = get_map_image(
                    source=self.tile_basemap, lon=geom.x, lat=geom.y
                )
                
                # Create image sized to fit panel
                tile_image = ipyw.Image(
                    value=image_bytes, format="png", width=115, height=115
                )
                
                # Get point ID
                point_id = str(row["id"])
                
                # Create map/location button (first)
                map_button = Button(
                    description="",
                    icon="fa-map-marker",  # Font Awesome f3c5
                    layout=Layout(
                        width="35px",
                        height="30px",
                        margin="0px 2px",
                        padding="2px"
                    ),
                    tooltip="Click to center map on this location"
                )
                
                # Create tick (positive) button
                tick_button = Button(
                    description="",
                    icon="fa-check",  # Font Awesome f00c
                    layout=Layout(
                        width="35px",
                        height="30px",
                        margin="0px 2px",
                        padding="2px"
                    ),
                    button_style="primary" if point_id in self.pos_ids else "",
                    tooltip="Click to label as positive"
                )
                tick_button.layout.opacity = "1.0" if point_id in self.pos_ids else "0.3"
                
                # Create cross (negative) button
                cross_button = Button(
                    description="",
                    icon="fa-times",  # Font Awesome f00d
                    layout=Layout(
                        width="35px",
                        height="30px",
                        margin="0px 2px",
                        padding="2px"
                    ),
                    button_style="warning" if point_id in self.neg_ids else "",
                    tooltip="Click to label as negative"
                )
                cross_button.layout.opacity = "1.0" if point_id in self.neg_ids else "0.3"
                
                # Store data for click handlers
                tick_button.point_id = point_id
                tick_button.row_data = row
                tick_button.is_positive = True
                tick_button.partner_button = cross_button
                
                cross_button.point_id = point_id
                cross_button.row_data = row
                cross_button.is_positive = False
                cross_button.partner_button = tick_button
                
                map_button.point_id = point_id
                map_button.row_data = row
                
                # Add click handlers
                tick_button.on_click(self._on_tile_label_click)
                cross_button.on_click(self._on_tile_label_click)
                map_button.on_click(self._on_tile_map_click)
                
                # Create button row with map button first
                button_row = HBox(
                    [map_button, tick_button, cross_button],
                    layout=Layout(
                        width="115px",  # Match image width for perfect centering
                        justify_content="center", 
                        margin="0 0 5px 0",
                        align_self="center"
                    )
                )
                
                # Create tile container with image
                tile_container = VBox(
                    [button_row, tile_image],
                    layout=Layout(
                        width="120px",  # Container sized for smaller images
                        padding="2px",
                        margin="0px",
                        align_items="center"  # Center all children
                    )
                )
                
                # Update the specific tile in the grid
                tiles_list = list(self.results_grid.children)
                if idx < len(tiles_list):
                    tiles_list[idx] = tile_container
                    self.results_grid.children = tiles_list
            except Exception as e:
                if self.verbose:
                    print(f"Error creating tile for result: {e}")
                # Replace loading with error placeholder
                error_label = ipyw.Label(
                    value="Error",
                    layout=ipyw.Layout(
                        width="120px",  # Match container width
                        height="155px",  # Height for smaller image + buttons
                        border="1px solid #ff0000",
                        display="flex",
                        align_items="center",
                        justify_content="center"
                    )
                )
                tiles_list = list(self.results_grid.children)
                if idx < len(tiles_list):
                    tiles_list[idx] = error_label
                    self.results_grid.children = tiles_list

        # Load tiles asynchronously
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, (_, row) in enumerate(all_displayed_df.iterrows()):
                if idx < current_tile_count:  # Only reload tiles that were displayed
                    future = executor.submit(create_and_update_tile, idx, row)
                    futures.append(future)
            
            # Wait for all tiles to complete
            for future in futures:
                future.result()

    def _on_next_tiles_click(self, b):
        """Handle next tiles button click."""
        self._show_operation_status("⏳ Loading next 50 tiles...")
        self.tile_page += 1
        self._update_results_panel(self.last_search_results_df)

    def _update_results_panel(self, search_results_df):
        """Loads 8 tiles initially, then pages of 50. All synchronous."""
        if search_results_df is None or search_results_df.empty:
            self.results_grid.children = []
            return

        def create_tile_widget(row):
            geom = shapely.wkt.loads(row["geometry_wkt"])
            image_bytes = get_map_image(source=self.tile_basemap, lon=geom.x, lat=geom.y)
            tile_image = ipyw.Image(value=image_bytes, format="png", width=115, height=115)
            point_id = str(row["id"])
            
            map_button = Button(icon="fa-map-marker", layout=Layout(width="35px", height="30px", margin="0px 2px", padding="2px"), tooltip="Center map")
            tick_button = Button(icon="fa-check", layout=Layout(width="35px", height="30px", margin="0px 2px", padding="2px"), button_style="primary" if point_id in self.pos_ids else "", tooltip="Label as positive")
            tick_button.layout.opacity = "1.0" if point_id in self.pos_ids else "0.3"
            cross_button = Button(icon="fa-times", layout=Layout(width="35px", height="30px", margin="0px 2px", padding="2px"), button_style="danger" if point_id in self.neg_ids else "", tooltip="Label as negative")
            cross_button.layout.opacity = "1.0" if point_id in self.neg_ids else "0.3"

            map_button.on_click(lambda b, g=geom: self._on_center_map_click(g))
            tick_button.on_click(lambda b, p=point_id, r=row, t=tick_button, c=cross_button: self._on_label_click(p, r, "pos", t, c))
            cross_button.on_click(lambda b, p=point_id, r=row, t=tick_button, c=cross_button: self._on_label_click(p, r, "neg", t, c))

            buttons = HBox([map_button, tick_button, cross_button], layout=Layout(justify_content="center"))
            return VBox([tile_image, buttons], layout=Layout(border="1px solid #ccc", padding="2px", width="120px", height="155px"))

        if self.tile_page == 0:
            self.results_grid.children = []
            page_df = search_results_df.head(self.initial_load_size)
            end_index = self.initial_load_size
            self.is_first_page_view = True
        else:
            start_index = self.initial_load_size + (self.tile_page - 1) * self.tiles_per_page
            end_index = start_index + self.tiles_per_page
            page_df = search_results_df.iloc[start_index:end_index]
        
        if page_df.empty:
            self.next_tiles_btn.layout.display = 'none'
            return

        with ThreadPoolExecutor(max_workers=8) as executor:
            new_tiles = list(executor.map(create_tile_widget, [row for _, row in page_df.iterrows()]))
        
        self.results_grid.children = tuple(list(self.results_grid.children) + new_tiles)

        if end_index < len(search_results_df):
            self.next_tiles_btn.layout.display = 'flex'
        else:
            self.next_tiles_btn.layout.display = 'none'

        self.tiles_button.button_style = 'success'
        self._show_operation_status("✅ Tiles loaded!")
        
    def _update_toggle_button_styles(self):
        """Update toggle button colors based on selection."""
        style = """
        <style>
        .widget-toggle-buttons button:nth-child(1).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        .widget-toggle-buttons button:nth-child(2).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        .widget-toggle-buttons button:nth-child(3).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        </style>
        """ % (UIConstants.POS_COLOR, UIConstants.NEG_COLOR, UIConstants.NEUTRAL_COLOR)
        display(HTML(style))

    def _add_map_layers(self):
        """Add all necessary layers to the map."""
        # Region boundary (optional)
        if hasattr(self, "effective_boundary_path") and self.effective_boundary_path:
            try:
                with open(self.effective_boundary_path) as f:
                    region_layer = ipyl.GeoJSON(
                        name="region",
                        data=json.load(f),
                        style=LayerStyles.get_region_style(),
                    )
                self.map.add_layer(region_layer)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not add boundary layer: {e}")

        # Positive layer
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.POS_COLOR),
        )
        self.map.add_layer(self.pos_layer)

        # Negative layer
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.NEG_COLOR),
        )
        self.map.add_layer(self.neg_layer)

        # Erase layer
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_erase_style(),
        )
        self.map.add_layer(self.erase_layer)

        # Points layer for search results
        self.points = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_search_style(),
            hover_style=LayerStyles.get_search_hover_style(),
        )
        self.map.add_layer(self.points)

    def _setup_draw_control(self):
        """Set up the draw control for lasso selection."""
        self.draw_control = DrawControl(
            polygon=LayerStyles.get_draw_options(),
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        self.draw_control.clear()

        # Track polygon drawing state
        self.polygon_drawing = False

    def _wire_events(self):
        """Wire all event handlers."""
        # Search button (main functionality)
        self.search_btn.on_click(self.search_click)

        # Reset button
        self.reset_btn.on_click(self.reset_all)

        # Tiles button
        self.tiles_button.on_click(self._on_tiles_click)

        # Label toggle
        self.label_toggle.observe(self._on_label_change, "value")

        # Selection mode toggle
        self.selection_mode.observe(self._on_selection_mode_change, "value")

        # # Neighbors slider
        # self.neighbors_slider.observe(self._on_neighbors_change, 'value')

        # Basemap buttons
        for basemap_name, btn in self.basemap_buttons.items():
            btn.on_click(lambda b, name=basemap_name: self._on_basemap_select(name))

        # Database dropdown
        if self.database_dropdown:
            self.database_dropdown.observe(self._on_database_change, names=["value"])

        # Collapse button
        self.collapse_btn.on_click(self._on_toggle_collapse)

        # Export and external tools
        self.save_btn.on_click(self.save_dataset)
        self.load_btn.on_click(self._on_load_click)
        self.file_upload.observe(self._on_file_upload, names=["value"])
        self.add_vector_btn.on_click(self._on_add_vector_click)
        self.vector_file_upload.observe(self._on_vector_file_upload, names=["value"])
        self.google_maps_btn.on_click(self._on_google_maps_click)
        self.run_button.on_click(self.search_click)

        # Map interactions
        self.map.on_interaction(self._on_map_interaction)

    def _on_label_change(self, change):
        """Handle label toggle change."""
        self.current_label = change["new"]
        if self.current_label == "Positive":
            self.select_val = UIConstants.POSITIVE_LABEL
        elif self.current_label == "Negative":
            self.select_val = UIConstants.NEGATIVE_LABEL
        else:  # Erase
            self.select_val = UIConstants.ERASE_LABEL
        self._update_status()

    def _on_google_maps_click(self, b):
        """Open current map center in Google Maps."""
        center = self.map.center
        url = f"https://www.google.com/maps/@{center[0]},{center[1]},15z"
        webbrowser.open(url, new=2)

    def _on_load_click(self, b):
        """Handle load dataset button click."""
        # Toggle file upload widget visibility
        if self.file_upload.layout.display == "none":
            self.file_upload.layout.display = "flex"
            self.load_btn.description = "📂 Cancel Load"
        else:
            self.file_upload.layout.display = "none"
            self.load_btn.description = "📂 Load Dataset"
            # Clear any uploaded files
            self.file_upload.value = ()

    def _on_file_upload(self, change):
        """Handle file upload."""
        if not change["new"]:
            return

        # Get the uploaded file - change['new'] is a tuple of uploaded files
        uploaded_files = change["new"]
        if not uploaded_files:
            return

        # Get the first uploaded file
        uploaded_file = uploaded_files[0]
        filename = uploaded_file["name"]
        content = uploaded_file["content"]

        try:
            self.load_dataset_from_content(content, filename)
            # Hide the upload widget and reset button text
            self.file_upload.layout.display = "none"
            self.load_btn.description = "📂 Load Dataset"
            # Clear the upload widget
            self.file_upload.value = ()
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            # Still hide the widget on error
            self.file_upload.layout.display = "none"
            self.load_btn.description = "📂 Load Dataset"

    def _on_add_vector_click(self, b):
        """Handle add vector layer button click."""
        # Toggle file upload widget visibility
        if self.vector_file_upload.layout.display == "none":
            self.vector_file_upload.layout.display = "flex"
            self.add_vector_btn.description = "📄 Cancel Vector"
        else:
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.description = "📄 Add Vector Layer"
            # Clear any uploaded files
            self.vector_file_upload.value = ()

    def _on_vector_file_upload(self, change):
        """Handle vector file upload."""
        if not change["new"]:
            return

        # Get the uploaded file - change['new'] is a tuple of uploaded files
        uploaded_files = change["new"]
        if not uploaded_files:
            return

        # Get the first uploaded file
        uploaded_file = uploaded_files[0]
        filename = uploaded_file["name"]
        content = uploaded_file["content"]

        try:
            self._add_vector_layer_from_content(content, filename)
            # Hide the upload widget and reset button text
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.description = "📄 Add Vector Layer"
            # Clear the upload widget
            self.vector_file_upload.value = ()
        except Exception as e:
            print(f"❌ Error loading vector file: {str(e)}")
            # Still hide the widget on error
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.description = "📄 Add Vector Layer"

    def _on_basemap_select(self, basemap_name):
        """Handle basemap selection."""
        self.current_basemap = basemap_name
        # Use instance basemap_tiles which includes EE basemaps
        if hasattr(self, "basemap_tiles"):
            self.basemap_layer.url = self.basemap_tiles[basemap_name]
        else:
            self.basemap_layer.url = BasemapConfig.BASEMAP_TILES[basemap_name]
        self._update_basemap_button_styles()

    def _on_toggle_collapse(self, b):
        """Toggle panel collapse/expand."""
        if self.panel_collapsed:
            # Expand
            self.accordion_container.layout.display = "flex"
            self.collapse_btn.description = "◀"
            self.panel_collapsed = False
        else:
            # Collapse
            self.accordion_container.layout.display = "none"
            self.collapse_btn.description = "▶"
            self.panel_collapsed = True

    def _on_map_interaction(self, **kwargs):
        """Handle all map interactions."""
        _log_to_file("Entered _on_map_interaction")
        lat, lon = kwargs.get('coordinates', (0, 0))
        
        # Update status
        self._update_status(lat, lon)

        # Handle shift-click for polygon drawing hint
        if kwargs.get("type") == "mousemove" and kwargs.get("modifiers", {}).get(
            "shiftKey", False
        ):
            self.status_bar.value += (
                " | <b>Hold Shift + Draw to select multiple points</b>"
            )

        # Handle ctrl-click for Google Maps
        if kwargs.get("type") == "click" and kwargs.get("modifiers", {}).get(
            "ctrlKey", False
        ):
            url = f"https://www.google.com/maps/@{lat},{lon},18z"
            webbrowser.open(url, new=2)
            _log_to_file("Handled as Ctrl-Click for Google Maps. Returning.")
            return

        # Normal label point behavior
        _log_to_file("Proceeding to label_point.")
        self.label_point(**kwargs)

    def _on_selection_mode_change(self, change):
        """Handle selection mode change."""
        self.lasso_mode = change["new"] == "polygon"
        self._update_status()

    def handle_draw(self, target, action, geo_json):
        """Handle polygon drawing with chunked embedding fetching."""
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            # Mark that we're processing a polygon
            self.polygon_drawing = False

            # Get the polygon geometry from the drawn shape and convert to shapely Polygon
            polygon_coords = geo_json["geometry"]["coordinates"][0]
            polygon = shapely.geometry.Polygon(polygon_coords)

            point_ids = []

            # First check cached detections
            if (
                self.detections_with_embeddings is not None
                and len(self.detections_with_embeddings) > 0
            ):
                # Find points within polygon from cached detections
                within_mask = self.detections_with_embeddings.geometry.within(polygon)
                cached_points = self.detections_with_embeddings[within_mask]

                point_ids.extend(cached_points["id"].tolist())

            # If no cached results or need more points, query the database
            if len(point_ids) == 0:
                polygon_wkt = polygon.wkt

                # Use lightweight query without embeddings
                points_in_polygon_query = f"""
                SELECT id
                FROM geo_embeddings
                WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
                """

                arrow_table = self.duckdb_connection.execute(
                    points_in_polygon_query
                ).fetch_arrow_table()
                points_inside = arrow_table.to_pandas()

                point_ids.extend(points_inside["id"].tolist())

            if not point_ids:
                if self.verbose:
                    print("⚠️ No points found within the selected polygon")
                self.draw_control.clear()
                self._update_status()
                return

            # Fetch embeddings in chunks for all points
            self._fetch_embeddings(point_ids)

            # Label all points (embeddings are now guaranteed to be cached)
            for point_id in point_ids:
                # Remove from existing labels
                if point_id in self.pos_ids:
                    self.pos_ids.remove(point_id)
                if point_id in self.neg_ids:
                    self.neg_ids.remove(point_id)

                # Add to appropriate label list
                if self.select_val == UIConstants.POSITIVE_LABEL:
                    self.pos_ids.append(point_id)
                elif self.select_val == UIConstants.NEGATIVE_LABEL:
                    self.neg_ids.append(point_id)

            # Show polygon labeling result in status bar
            self._show_operation_status(
                f"✅ Labeled {len(point_ids)} points as {self.current_label}"
            )
            if self.verbose:
                print(f"✅ Labeled {len(point_ids)} points as {self.current_label}")

            self.update_layers()
            self.update_query_vector()

            # Clear the polygon after processing
            self.draw_control.clear()
            self._update_status()

        elif action == "drawstart":
            # Mark that we're starting to draw a polygon
            if self.lasso_mode:
                self.polygon_drawing = True
                self._update_status()

        elif action == "deleted":
            # Reset polygon drawing state
            self.polygon_drawing = False
            self._update_status()

    def _update_status(self, lat=None, lon=None, operation_msg=None):
        """Update the status bar."""
        if lat is None or lon is None:
            center = self.map.center
            lat, lon = center[0], center[1]

        mode = "Polygon" if self.lasso_mode else "Point"
        label = self.current_label

        status_text = f"Lat: {lat:.4f} | Lon: {lon:.4f} | Mode: {mode} | Label: {label}"

        if self.lasso_mode:
            if self.polygon_drawing:
                status_text += " | <b>Drawing polygon...</b>"

        # Add operation message if provided, otherwise use current operation
        display_operation = operation_msg or self.current_operation
        if display_operation:
            status_text += f"<br/><span style='color: #0072B2; font-weight: bold;'>{display_operation}</span>"

        self.status_bar.value = f"""
            <div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>
                {status_text}
            </div>
        """

    def _show_operation_status(self, message):
        """Show an operation status message in the status bar."""
        self.current_operation = message
        self._update_status(operation_msg=message)

    def _clear_operation_status(self):
        """Clear the current operation status."""
        self.current_operation = None
        self._update_status()

    def _prepare_ids_for_query(self, id_list):
        """Prepare IDs for database queries, handling both string and integer IDs.

        Args:
            id_list: List of ID values (strings or integers)

        Returns:
            List of values appropriate for database queries
        """
        # Return IDs as-is since DuckDB can handle both strings and integers
        return [str(id_val) for id_val in id_list]

    def reset_all(self, b):
        """Reset all labels, search results, and cached data."""
        if self.verbose:
            print("🗑️ Resetting all labels and search results...")

        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}
        self.query_vector = None
        self.detections_with_embeddings = None

        empty_geojson = {"type": "FeatureCollection", "features": []}
        self.pos_layer.data = empty_geojson
        self.neg_layer.data = empty_geojson
        self.erase_layer.data = empty_geojson
        self.points.data = empty_geojson

        if self.vector_layer:
            if self.vector_layer in self.map.layers:
                self.map.remove_layer(self.vector_layer)
            self.vector_layer = None
        
        for layer in self.map.layers:
            if hasattr(layer, 'name') and layer.name == 'tile_highlight':
                self.map.remove_layer(layer)

        self.results_grid.children = []
        self.tiles_pane.layout.display = 'none'
        self.tiles_button.button_style = ''

        self._clear_operation_status()

        if self.verbose:
            print("✅ All data cleared!")

    def _fetch_embeddings(self, point_ids):
        """Fetch embeddings for given point IDs in chunks and cache them."""
        if not point_ids:
            return

        # Show progress for large batches
        if len(point_ids) > 100:
            self._show_operation_status(
                f"🔄 Fetching embeddings for {len(point_ids)} points..."
            )
            if self.verbose:
                print(f"🔄 Fetching embeddings for {len(point_ids)} points...")

        # Process in chunks to avoid memory issues
        for i in range(0, len(point_ids), DatabaseConstants.EMBEDDING_CHUNK_SIZE):
            chunk = point_ids[i : i + DatabaseConstants.EMBEDDING_CHUNK_SIZE]

            # Show chunk progress for very large batches
            if len(point_ids) > DatabaseConstants.EMBEDDING_CHUNK_SIZE:
                chunk_num = i // DatabaseConstants.EMBEDDING_CHUNK_SIZE + 1
                total_chunks = (len(point_ids) - 1) // DatabaseConstants.EMBEDDING_CHUNK_SIZE + 1
                self._show_operation_status(
                    f"🔄 Processing chunk {chunk_num}/{total_chunks}"
                )
                if self.verbose:
                    print(
                        f"   Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} points)..."
                    )

            # Build parameterized query for this chunk
            prepared_chunk = self._prepare_ids_for_query(chunk)
            placeholders = ",".join(["?" for _ in prepared_chunk])
            query = f"""
            SELECT id, tile_id, CAST(embedding AS FLOAT[]) as embedding, geometry 
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            
            _log_to_file(f"Fetch embeddings: Built query for chunk with IDs: {prepared_chunk}")
            if self.verbose:
                print(f"DEBUG: Executing embedding fetch query for IDs: {prepared_chunk}")
            
            # Fetch as Arrow then convert to pandas
            _log_to_file("Fetch embeddings: About to execute query.")
            arrow_table = self.duckdb_connection.execute(query, prepared_chunk).fetch_arrow_table()
            _log_to_file("Fetch embeddings: Query executed successfully. About to fetch arrow table.")
            
            if self.verbose:
                print(f"DEBUG: Successfully executed embedding fetch query. Processing results.")

            chunk_df = arrow_table.to_pandas()
            _log_to_file("Fetch embeddings: Converted arrow table to pandas.")
            
            # Cache the embeddings from this chunk
            for _, row in chunk_df.iterrows():
                embedding = np.array(row["embedding"])
                # Ensure consistent string type for point IDs
                point_id = str(row["id"])
                self.cached_embeddings[point_id] = embedding

        if len(point_ids) > 100:
            self._show_operation_status(
                f"✅ Cached embeddings for {len(point_ids)} points"
            )
            if self.verbose:
                print(f"✅ Cached embeddings for {len(point_ids)} points")

    def search_click(self, b):
        """Handle the main search button click event."""
        self.tile_page = 0
        if self.query_vector is None or len(self.query_vector) == 0:
            if self.verbose:
                print("🔍 No query vector. Please label some points first.")
            return

        self._search_faiss()

    def _search_faiss(self):
        """Perform similarity search using the loaded FAISS index."""
        if self.verbose:
            print("🧠 Performing search with FAISS index...")
        
        n_neighbors = self.neighbors_slider.value
        all_labeled_ids = self.pos_ids + self.neg_ids
        extra_results = min(len(all_labeled_ids), n_neighbors // 2)
        total_requested = n_neighbors + extra_results

        # Step A: Query FAISS
        # TODO: make nprobe dynamic based on number of labeled points, or allow user input
        params = faiss.SearchParametersIVF(nprobe=4096)
        query_vector_np = self.query_vector.reshape(1, -1).astype('float32')
        self._show_operation_status(f"🔍 FAISS Search: Finding {n_neighbors} neighbors...")
        distances, ids = self.faiss_index.search(query_vector_np, total_requested, params=params)
        faiss_ids = ids[0].tolist()
        faiss_distances = distances[0].tolist()

        # Step B: Query DuckDB for metadata
        if not faiss_ids:
            if self.verbose:
                print("⚠️ FAISS search returned no results.")
            self._process_and_display_search_results(pd.DataFrame(), n_neighbors)
            return

        placeholders = ','.join(['?' for _ in faiss_ids])
        sql = f"""
        SELECT id, ST_AsGeoJSON(geometry) AS geometry_json, ST_AsText(geometry) AS geometry_wkt
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        
        # Preserve order of FAISS results
        id_map = {id_val: i for i, id_val in enumerate(faiss_ids)}
        
        # Fetch as pandas DataFrame
        metadata_df = self.duckdb_connection.execute(sql, faiss_ids).fetchdf()
        
        # Sort results according to FAISS distance and add distance column
        metadata_df['sort_order'] = metadata_df['id'].map(id_map)
        metadata_df = metadata_df.sort_values('sort_order').drop(columns=['sort_order'])
        metadata_df['distance'] = faiss_distances[:len(metadata_df)]
        
        self._process_and_display_search_results(metadata_df, n_neighbors)

    def _process_and_display_search_results(self, search_results_df, n_neighbors):
        """Filters, processes, and displays search results on the map."""
        all_labeled_ids = self.pos_ids + self.neg_ids

        if search_results_df.empty:
            self._show_operation_status("✅ Search complete. No results found.")
            self.points.data = {"type": "FeatureCollection", "features": []}
            return

        # Post-filter to exclude labeled points
        if all_labeled_ids:
            labeled_id_strings = set(str(lid) for lid in all_labeled_ids)
            mask = ~search_results_df['id'].astype(str).isin(labeled_id_strings)
            search_results_filtered = search_results_df[mask].head(n_neighbors)
        else:
            search_results_filtered = search_results_df.head(n_neighbors)
        
        filtered_count = len(search_results_filtered)
        self._show_operation_status(f"✅ Found {filtered_count} similar points.")
        if self.verbose:
            print(f"✅ Found {filtered_count} similar points after filtering.")
        
        geometries = [shapely.wkt.loads(row['geometry_wkt']) for _, row in search_results_filtered.iterrows()]
        
        self.detections_with_embeddings = gpd.GeoDataFrame({
            'id': search_results_filtered['id'].astype(str).values,
            'distance': search_results_filtered['distance'].values,
            'geometry': geometries
        })
        
        detections_geojson = {"type": "FeatureCollection", "features": []}
        if not search_results_filtered.empty:
            min_distance = search_results_filtered['distance'].min()
            max_distance = search_results_filtered['distance'].max()
            
            # Sort by distance in descending order so most similar (green) points render last and appear on top
            search_results_sorted = search_results_filtered.sort_values('distance', ascending=False)
            
            for _, row in search_results_sorted.iterrows():
                color = UIConstants.distance_to_color(row['distance'], min_distance, max_distance)
                detections_geojson["features"].append({
                    "type": "Feature",
                    "geometry": json.loads(row['geometry_json']),
                    "properties": {
                        "id": str(row['id']),
                        "distance": row['distance'],
                        "color": color,
                        "fillColor": color
                    }
                })
        
        self.last_search_results_df = search_results_filtered.copy()
        self.tile_page = 0
        
        self._update_search_layer_with_colors(detections_geojson)
        self._update_results_panel(self.last_search_results_df)

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        # Don't process clicks when in polygon mode or actively drawing
        if not self.execute_label_point or self.lasso_mode or self.polygon_drawing:
            return

        action = kwargs.get("type")
        if action not in ["click"]:
            return
                 
        lat, lon = kwargs.get('coordinates')
        
        # Query the database for the single nearest point to the click
        _log_to_file("label_point: Querying database for nearest point.")
        
        sql = DatabaseConstants.NEAREST_POINT_QUERY
        params = [lon, lat]
        result = self.duckdb_connection.execute(sql, params).fetchone()
        
        if result is None:
            self._show_operation_status("⚠️ No points found near click.")
            _log_to_file("label_point: No point found. Returning.")
            return
        
        point_id = str(result[0])
        embedding = np.array(result[3])
        self.cached_embeddings[point_id] = embedding # Cache the embedding
        
        if self.verbose:
            print(f"DEBUG: Found point ID: {point_id}.")

        # Update labels
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)

        if self.select_val == UIConstants.POSITIVE_LABEL:
            self.pos_ids.append(point_id)
        elif self.select_val == UIConstants.NEGATIVE_LABEL:
            self.neg_ids.append(point_id)
        else:
            # For erase mode, get the point geometry from DuckDB
            erase_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id = ?
            """
            # Prepare ID for database query
            prepared_point_id = str(point_id)
            erase_result = self.duckdb_connection.execute(
                erase_query, [prepared_point_id]
            ).fetchone()
            if erase_result:
                erase_geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": json.loads(erase_result[0]),
                            "properties": {},
                        }
                    ],
                }
                self.erase_layer.data = erase_geojson

        # Update visualization and query vector immediately
        self.update_layers()
        self.update_query_vector()

    def update_layer(self, layer, geojson_data):
        """Update a specific layer with new GeoJSON data."""
        layer.data = geojson_data

    def _update_search_layer_with_colors(self, geojson_data: Dict) -> None:
        """Update search results layer with distance-based coloring."""

        # Define style function that uses per-feature colors
        def style_function(feature):
            props = feature.get("properties", {})
            return {
                "color": "black",
                "radius": UIConstants.SEARCH_POINT_RADIUS,
                "fillColor": props.get("fillColor", UIConstants.SEARCH_COLOR),
                "opacity": UIConstants.POINT_OPACITY,
                "fillOpacity": UIConstants.POINT_FILL_OPACITY,
                "weight": UIConstants.SEARCH_POINT_WEIGHT,
            }

        # Update the points layer with new data and style function
        self.points.data = geojson_data
        self.points.style_callback = style_function

    def update_layers(self):
        if self.pos_ids:
            # Prepare IDs for database query
            prepared_pos_ids = self._prepare_ids_for_query(self.pos_ids)
            placeholders = ",".join(["?" for _ in prepared_pos_ids])
            pos_query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            pos_results = self.duckdb_connection.execute(
                pos_query, prepared_pos_ids
            ).df()
            pos_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in pos_results.iterrows():
                pos_geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(row["geometry"]),
                        "properties": {},
                    }
                )
            self.pos_layer.data = pos_geojson
        else:
            self.pos_layer.data = {"type": "FeatureCollection", "features": []}

        if self.neg_ids:
            # Prepare IDs for database query
            prepared_neg_ids = self._prepare_ids_for_query(self.neg_ids)
            placeholders = ",".join(["?" for _ in prepared_neg_ids])
            neg_query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            neg_results = self.duckdb_connection.execute(
                neg_query, prepared_neg_ids
            ).df()
            neg_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in neg_results.iterrows():
                neg_geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(row["geometry"]),
                        "properties": {},
                    }
                )
            self.neg_layer.data = neg_geojson
        else:
            self.neg_layer.data = {"type": "FeatureCollection", "features": []}

    def update_query_vector(self, skip_fetch=False):
        """Update the query vector based on current positive and negative labels.

        Args:
            skip_fetch: If True, assume embeddings are already cached (optimization for single-point labeling)
        """
        if not self.pos_ids:
            self.query_vector = None
            return

        # Only fetch missing embeddings if not skipping (for efficiency in single-point labeling)
        if not skip_fetch:
            # Fetch missing embeddings for positive labels using chunked method
            self._fetch_embeddings(self.pos_ids)

            # Fetch missing embeddings for negative labels using chunked method
            if self.neg_ids:
                self._fetch_embeddings(self.neg_ids)

        # Get positive embeddings from cache
        pos_embeddings = [
            self.cached_embeddings[pid]
            for pid in self.pos_ids
            if pid in self.cached_embeddings
        ]

        if not pos_embeddings:
            self.query_vector = None
            return

        pos_vec = np.mean(pos_embeddings, axis=0)

        # Get negative embeddings from cache
        neg_embeddings = [
            self.cached_embeddings[nid]
            for nid in self.neg_ids
            if nid in self.cached_embeddings
        ]

        if neg_embeddings:
            neg_vec = np.mean(neg_embeddings, axis=0)
        else:
            neg_vec = np.zeros_like(pos_vec)

        # Default query vector math
        self.query_vector = 2 * pos_vec - neg_vec

    def save_dataset(self, b):
        """Save labeled points with embeddings to a GeoJSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if we have any labels to save
        if not self.pos_ids and not self.neg_ids:
            if self.verbose:
                print("⚠️ No labeled points to save.")
            return

        if self.verbose:
            print("💾 Saving dataset...")

        # Combine all labeled IDs
        all_labeled_ids = list(set(self.pos_ids + self.neg_ids))

        if not all_labeled_ids:
            if self.verbose:
                print("⚠️ No valid labels to save.")
            return

        # Query database for all labeled points with their geometries and embeddings
        # Prepare IDs for database query
        prepared_labeled_ids = self._prepare_ids_for_query(all_labeled_ids)
        placeholders = ",".join(["?" for _ in prepared_labeled_ids])
        query = f"""
        SELECT 
            id,
            ST_AsText(geometry) AS wkt,
            ST_AsGeoJSON(geometry) AS geometry_json,
            embedding
        FROM geo_embeddings 
        WHERE id IN ({placeholders})
        """

        results = self.duckdb_connection.execute(query, prepared_labeled_ids).df()

        if results.empty:
            if self.verbose:
                print("⚠️ Could not retrieve data for labeled points.")
            return

        # Create lists to store the data
        features = []

        # Process each result
        for _, row in results.iterrows():
            point_id = str(row["id"])  # Ensure string type for consistency

            # Determine label (positive or negative)
            if point_id in self.pos_ids:
                label = UIConstants.POSITIVE_LABEL
            elif point_id in self.neg_ids:
                label = UIConstants.NEGATIVE_LABEL
            else:
                continue  # Skip if somehow not in either list

            # Get embedding (from cache or from query result)
            if point_id in self.cached_embeddings:
                embedding = self.cached_embeddings[point_id]
            else:
                embedding = np.array(row["embedding"])

            # Create feature with properties including label and embedding
            feature = {
                "type": "Feature",
                "geometry": json.loads(row["geometry_json"]),
                "properties": {
                    "id": point_id,
                    "label": label,
                    "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
                },
            }
            features.append(feature)

        # Create GeoJSON structure
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

        # Save to file
        filename = f"labeled_dataset_{timestamp}.geojson"

        try:
            with open(filename, "w") as f:
                json.dump(geojson_data, f, indent=2)

            # Create summary
            pos_count = len(
                [
                    f
                    for f in features
                    if f["properties"]["label"] == UIConstants.POSITIVE_LABEL
                ]
            )
            neg_count = len(
                [
                    f
                    for f in features
                    if f["properties"]["label"] == UIConstants.NEGATIVE_LABEL
                ]
            )

            if self.verbose:
                print("✅ Dataset saved successfully!")
                print(f"📄 Filename: {filename}")
                print("📊 Summary:")
                print(f"   - Total points: {len(features)}")
                print(f"   - Positive labels: {pos_count}")
                print(f"   - Negative labels: {neg_count}")
                print(f"   - Embedding dimension: {self.embedding_dim}")

            # Optional: Also save a separate CSV with just IDs and labels for easier processing
            labels_df = pd.DataFrame(
                [
                    {"id": f["properties"]["id"], "label": f["properties"]["label"]}
                    for f in features
                ]
            )
            csv_filename = f"labeled_dataset_{timestamp}_labels.csv"
            labels_df.to_csv(csv_filename, index=False)
            if self.verbose:
                print(f"📄 Also saved labels CSV: {csv_filename}")

        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving dataset: {str(e)}")

    def load_dataset(self, filename):
        """Load a previously saved labeled dataset."""
        try:
            with open(filename, "r") as f:
                geojson_data = json.load(f)

            # Clear current labels
            self.pos_ids = []
            self.neg_ids = []
            self.cached_embeddings = {}

            # Process features
            for feature in geojson_data["features"]:
                point_id = str(feature["properties"]["id"])  # Ensure string type
                label = feature["properties"]["label"]
                embedding = np.array(feature["properties"]["embedding"])

                # Cache the embedding
                self.cached_embeddings[point_id] = embedding

                # Add to appropriate list
                if label == UIConstants.POSITIVE_LABEL:
                    self.pos_ids.append(point_id)
                elif label == UIConstants.NEGATIVE_LABEL:
                    self.neg_ids.append(point_id)

            # Update visualization
            self.update_layers()
            self.update_query_vector()

            # Print summary
            metadata = geojson_data.get("metadata", {})
            if self.verbose:
                print("✅ Dataset loaded successfully!")
                print("📊 Summary:")
                print(
                    f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}"
                )
                print(f"   - Positive labels: {len(self.pos_ids)}")
                print(f"   - Negative labels: {len(self.neg_ids)}")
                print(f"   - Saved on: {metadata.get('timestamp', 'Unknown')}")

        except FileNotFoundError:
            if self.verbose:
                print(f"❌ File not found: {filename}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading dataset: {str(e)}")

    def load_dataset_from_content(self, content, filename):
        """Load a dataset from uploaded file content."""
        if self.verbose:
            print(f"📂 Loading dataset from {filename}...")

        try:
            # Convert content to bytes if it's a memoryview
            if isinstance(content, memoryview):
                content_bytes = content.tobytes()
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                content_bytes = bytes(content)

            # Determine file type and parse accordingly
            if filename.lower().endswith(".geojson"):
                # Parse GeoJSON
                geojson_data = json.loads(content_bytes.decode("utf-8"))
                self._process_geojson_data(geojson_data, filename)

            elif filename.lower().endswith(".parquet"):
                # Parse GeoParquet using pandas/geopandas
                import io

                gdf = gpd.read_parquet(io.BytesIO(content_bytes))
                self._process_geoparquet_data(gdf, filename)

            else:
                raise ValueError(
                    "Unsupported file format. Please use .geojson or .parquet files."
                )

        except Exception as e:
            raise Exception(f"Error processing {filename}: {str(e)}")

    def _process_geojson_data(self, geojson_data, filename):
        """Process GeoJSON data and populate labels."""
        # Clear current labels
        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}

        # Process features
        for feature in geojson_data["features"]:
            point_id = str(feature["properties"]["id"])  # Ensure string type
            label = feature["properties"]["label"]
            embedding = np.array(feature["properties"]["embedding"])

            # Cache the embedding
            self.cached_embeddings[point_id] = embedding

            # Add to appropriate list
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)

        # Update visualization
        self.update_layers()
        self.update_query_vector()

        # Print summary
        metadata = geojson_data.get("metadata", {})
        if self.verbose:
            print(f"✅ Dataset loaded successfully from {filename}!")
            print("📊 Summary:")
            print(
                f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}"
            )
            print(f"   - Positive labels: {len(self.pos_ids)}")
            print(f"   - Negative labels: {len(self.neg_ids)}")
            print(f"   - Saved on: {metadata.get('timestamp', 'Unknown')}")

    def _process_geoparquet_data(self, gdf, filename):
        """Process GeoParquet data and populate labels."""
        # Clear current labels
        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}

        # Check required columns
        required_cols = ["id", "label", "embedding"]
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Required column '{col}' not found in {filename}")

        # Process each row
        for _, row in gdf.iterrows():
            point_id = str(row["id"])  # Ensure string type
            label = row["label"]

            # Handle embedding - could be stored as array or list
            if isinstance(row["embedding"], (list, np.ndarray)):
                embedding = np.array(row["embedding"])
            else:
                # Try to parse if it's stored as string
                embedding = np.array(json.loads(row["embedding"]))

            # Cache the embedding
            self.cached_embeddings[point_id] = embedding

            # Add to appropriate list
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)

        # Update visualization
        self.update_layers()
        self.update_query_vector()

        # Print summary
        if self.verbose:
            print(f"✅ Dataset loaded successfully from {filename}!")
            print("📊 Summary:")
            print(f"   - Total points: {len(gdf)}")
            print(f"   - Positive labels: {len(self.pos_ids)}")
            print(f"   - Negative labels: {len(self.neg_ids)}")

    def _add_vector_layer_from_content(self, content, filename):
        """Add a vector layer from uploaded file content."""
        if self.verbose:
            print(f"📄 Adding vector layer from {filename}...")

        try:
            # Convert content to bytes if it's a memoryview
            if isinstance(content, memoryview):
                content_bytes = content.tobytes()
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                content_bytes = bytes(content)

            # Remove existing vector layer if it exists
            if self.vector_layer:
                if self.vector_layer in self.map.layers:
                    self.map.remove_layer(self.vector_layer)
                self.vector_layer = None

            # Determine file type and parse accordingly
            if filename.lower().endswith(".geojson"):
                # Parse GeoJSON
                geojson_data = json.loads(content_bytes.decode("utf-8"))
                self._add_vector_layer_from_geojson(geojson_data, filename)

            elif filename.lower().endswith(".parquet"):
                # Parse GeoParquet using geopandas
                import io

                gdf = gpd.read_parquet(io.BytesIO(content_bytes))
                self._add_vector_layer_from_geodataframe(gdf, filename)

            else:
                raise ValueError(
                    "Unsupported file format. Please use .geojson or .parquet files."
                )

        except Exception as e:
            raise Exception(f"Error processing vector file {filename}: {str(e)}")

    def _add_vector_layer_from_geojson(self, geojson_data, filename):
        """Add vector layer from GeoJSON data."""
        # Create vector layer with custom styling
        vector_style = {
            "color": "#FF6B6B",  # Red outline
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#FF6B6B",
            "fillOpacity": 0.3,
        }

        self.vector_layer = ipyl.GeoJSON(
            name=f"vector_layer_{filename}", data=geojson_data, style=vector_style
        )

        self.map.add_layer(self.vector_layer)

        # Print summary
        if self.verbose:
            feature_count = len(geojson_data.get("features", []))
            print(f"✅ Vector layer added successfully from {filename}!")
            print("📊 Summary:")
            print(f"   - Features: {feature_count}")

    def _add_vector_layer_from_geodataframe(self, gdf, filename):
        """Add vector layer from GeoDataFrame."""
        # Convert GeoDataFrame to GeoJSON
        geojson_data = json.loads(gdf.to_json())

        # Create vector layer with custom styling
        vector_style = {
            "color": "#FF6B6B",  # Red outline
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#FF6B6B",
            "fillOpacity": 0.3,
        }

        self.vector_layer = ipyl.GeoJSON(
            name=f"vector_layer_{filename}", data=geojson_data, style=vector_style
        )

        self.map.add_layer(self.vector_layer)

        # Print summary
        if self.verbose:
            print(f"✅ Vector layer added successfully from {filename}!")
            print("📊 Summary:")
            print(f"   - Features: {len(gdf)}")
            print(f"   - Geometry types: {gdf.geom_type.value_counts().to_dict()}")

    def _update_basemap_button_styles(self):
        """Update basemap button styles to highlight current selection."""
        for basemap_name, btn in self.basemap_buttons.items():
            if basemap_name == self.current_basemap:
                btn.button_style = "info"  # Blue highlight for active
            else:
                btn.button_style = ""  # Default style

    def _construct_boundary_path(self, database_path: str) -> str:
        """Construct boundary path from database path.

        Args:
            database_path: Path to database (e.g., gs://geovibes/databases/google/bali.db)

        Returns:
            Constructed boundary path (e.g., gs://geovibes/geometries/bali.geojson)
        """
        import os

        # Extract the database name without extension
        db_filename = os.path.basename(
            database_path
        )  # e.g., "bali.db" or "bali_google.db"
        db_name_with_ext = os.path.splitext(db_filename)[
            0
        ]  # e.g., "bali" or "bali_google"

        # For boundary files, we want just the region name (first part before underscore if any)
        # e.g., "bali_google" -> "bali", "java_google" -> "java"
        if "_" in db_name_with_ext:
            db_name = db_name_with_ext.split("_")[0]
        else:
            db_name = db_name_with_ext

        # Replace the databases part with geometries
        if database_path.startswith("gs://"):
            # Handle GCS paths
            parts = database_path.split("/")
            # Find the bucket and construct new path
            bucket = parts[2]  # e.g., "geovibes"
            boundary_path = f"gs://{bucket}/geometries/{db_name}.geojson"
        else:
            # Handle local paths
            db_dir = os.path.dirname(database_path)
            # Go up one level and enter geometries folder
            parent_dir = os.path.dirname(db_dir)
            boundary_path = os.path.join(parent_dir, "geometries", f"{db_name}.geojson")

        return boundary_path

    def _update_ee_boundary(self):
        """Update Earth Engine boundary based on current effective boundary path."""
        if not self.ee_available:
            return

        if self.effective_boundary_path:
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
        else:
            self.ee_boundary = None

    def _update_boundary_layer(self):
        """Update or add the boundary layer on the map."""
        # Remove existing boundary layer if present
        layers_to_remove = [
            layer
            for layer in self.map.layers
            if getattr(layer, "name", None) == "region"
        ]
        for layer in layers_to_remove:
            self.map.remove_layer(layer)

        # Add new boundary layer if we have an effective boundary path
        if hasattr(self, "effective_boundary_path") and self.effective_boundary_path:
            try:
                if self.verbose:
                    print(f"🗺️  Loading boundary layer: {self.effective_boundary_path}")

                # Use geopandas to read the file (handles both local and GCS paths)
                boundary_gdf = gpd.read_file(self.effective_boundary_path)

                # Convert to GeoJSON format for ipyleaflet
                boundary_geojson = boundary_gdf.to_json()

                region_layer = ipyl.GeoJSON(
                    name="region",
                    data=json.loads(boundary_geojson),
                    style=LayerStyles.get_region_style(),
                )
                self.map.add_layer(region_layer)
                if self.verbose:
                    print("✅ Boundary layer added successfully")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Could not add boundary layer: {e}")

    def _setup_boundary_and_center(self):
        """Set up boundary path and get map center coordinates.

        Returns:
            Tuple of (center_y, center_x) coordinates
        """
        # Determine boundary path
        boundary_path = self.config.boundary_path
        if not boundary_path and self.current_database_path:
            # Auto-construct boundary path from database path
            boundary_path = self._construct_boundary_path(self.current_database_path)

        if boundary_path:
            try:
                # Use boundary file for centering
                boundary_gdf = gpd.read_file(boundary_path)
                center_y, center_x = (
                    boundary_gdf.geometry.iloc[0].centroid.y,
                    boundary_gdf.geometry.iloc[0].centroid.x,
                )
                # Store the effective boundary path for later use
                self.effective_boundary_path = boundary_path
                if self.verbose:
                    print(f"📍 Using boundary file: {boundary_path}")
                return center_y, center_x
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not load boundary file {boundary_path}: {e}")
                    print("⚠️  Using database centroid for centering")
                # Fallback to database centroid
                self.effective_boundary_path = None
        else:
            self.effective_boundary_path = None

        # Use database centroid for centering
        center_y, center_x = get_database_centroid(
            self.duckdb_connection, verbose=self.verbose
        )
        return center_y, center_x

    def _warm_up_gcs_database(self):
        """Warm up GCS database with initial search for better performance."""
        try:
            if self.verbose:
                print("🔧 Optimizing database connection...")

            # Get the first point's embedding from the database
            first_point_query = """
            SELECT CAST(embedding AS FLOAT[]) as embedding 
            FROM geo_embeddings 
            WHERE embedding IS NOT NULL 
            LIMIT 1
            """

            result = self.duckdb_connection.execute(first_point_query).fetchone()
            if not result or not result[0]:
                if self.verbose:
                    print("⚠️  No embeddings found for warm-up")
                return

            first_embedding = result[0]

            # Run a similarity search with 100 neighbors to warm up the database
            sql = DatabaseConstants.get_similarity_search_light_query(
                self.embedding_dim
            )
            query_params = [first_embedding, 100]

            # Execute the warm-up query
            self.duckdb_connection.execute(sql, query_params).fetchall()

            if self.verbose:
                print("✅ Database optimization completed")

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Database warm-up failed: {str(e)}")

    def _on_database_change(self, change):
        """Handle database selection change."""
        new_database_path = change["new"]

        if new_database_path == self.current_database_path:
            return  # No change

        if self.verbose:
            print(f"🔄 Switching to database: {os.path.basename(new_database_path)}")

        # Show loading message immediately
        self._show_operation_status(
            "🔄 Loading database (this can take a couple of seconds)..."
        )

        try:
            # Step 1: Quick UI updates - pan map and switch boundary first
            old_database_path = self.current_database_path
            self.current_database_path = new_database_path

            # Update boundary path and recenter map immediately (fast operation)
            lat, lon = (
                self._setup_boundary_and_center()
            )  # Returns (lat, lon) - center_y, center_x
            self.map.center = (lat, lon)  # ipyleaflet expects (lat, lon)

            if self.verbose:
                print(f"📍 Map recentered to: {lat:.4f}, {lon:.4f}")

            # Update Earth Engine boundary (also relatively fast)
            self._update_ee_boundary()

            # Update boundary layer on map
            self._update_boundary_layer()

            # Step 2: Heavy database operations in background
            self._show_operation_status("🔄 Connecting to new database...")

            # Close current connection if we own it
            if hasattr(self, "_owns_connection") and self._owns_connection:
                if hasattr(self, "duckdb_connection") and self.duckdb_connection:
                    self.duckdb_connection.close()

            # Reset all application state
            self._reset_all_state()

            # Establish new connection
            self.duckdb_connection = DatabaseConstants.setup_duckdb_connection(
                new_database_path, read_only=True
            )
            self._owns_connection = True
            
            # Configure memory limits and disable progress bar
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)
            
            # Extra insurance: explicitly disable progress bar again
            try:
                self.duckdb_connection.execute("SET enable_progress_bar=false")
                self.duckdb_connection.execute("SET enable_profiling=false")
                self.duckdb_connection.execute("SET enable_object_cache=false")
            except:
                pass  # Ignore if these settings don't exist
            
            # Setup extensions
            if new_database_path:
                extension_queries = DatabaseConstants.get_extension_setup_queries(new_database_path)
                for query in extension_queries:
                    self.duckdb_connection.execute(query)

            # Load new FAISS index
            new_faiss_path = [
                db["faiss_path"]
                for db in self.available_databases
                if db["db_path"] == new_database_path
            ][0]
            self.current_faiss_path = new_faiss_path
            self._show_operation_status("🔄 Loading search index...")
            self.faiss_index = faiss.read_index(self.current_faiss_path)
            
            # Detect embedding dimension
            self._show_operation_status("🔄 Analyzing database structure...")
            try:
                self.embedding_dim = DatabaseConstants.detect_embedding_dimension(
                    self.duckdb_connection
                )
                if self.verbose:
                    print(f"🔍 Detected embedding dimension: {self.embedding_dim}")
            except ValueError as e:
                if self.verbose:
                    print(f"⚠️ Could not detect embedding dimension: {e}")
                self.embedding_dim = 384

            # Warm up GCS database if needed
            if DatabaseConstants.is_gcs_path(new_database_path):
                self._show_operation_status("🔄 Optimizing database connection...")
                self._warm_up_gcs_database()

            # Success!
            self._show_operation_status(
                f"✅ Successfully loaded: {os.path.basename(new_database_path)}"
            )
            if self.verbose:
                print(
                    f"✅ Successfully switched to database: {os.path.basename(new_database_path)}"
                )

        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch database: {str(e)}")

            # Revert to previous database
            self.current_database_path = old_database_path
            if self.database_dropdown:
                self.database_dropdown.value = old_database_path

            self._show_operation_status(f"❌ Failed to load database: {str(e)}")

            # Try to restore previous map position
            try:
                lat, lon = self._setup_boundary_and_center()
                self.map.center = (lat, lon)
            except:
                pass  # If this fails too, just leave the map where it is

    def _reset_all_state(self):
        """Reset all application state for database switching."""
        # Clear labels
        self.pos_ids = []
        self.neg_ids = []

        # Clear cached data
        self.cached_embeddings = {}
        self.detections_with_embeddings = None
        self.query_vector = None

        # Clear map layers
        empty_geojson = {"type": "FeatureCollection", "features": []}
        self.pos_layer.data = empty_geojson
        self.neg_layer.data = empty_geojson
        self.erase_layer.data = empty_geojson
        self.points.data = empty_geojson

        # Remove vector layer if it exists
        if self.vector_layer:
            if self.vector_layer in self.map.layers:
                self.map.remove_layer(self.vector_layer)
            self.vector_layer = None

        # Clear results panel and reset tiles state
        self.results_grid.children = []
        self.tiles_pane.layout.display = 'none'
        self.tiles_button.button_style = ''
        self.last_search_results_df = None
        self.tile_page = 0

        # Clear operation status
        self._clear_operation_status()

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_owns_connection") and self._owns_connection:
            if hasattr(self, "duckdb_connection") and self.duckdb_connection:
                self.duckdb_connection.close()
                if self.verbose:
                    print("🔌 DuckDB connection closed.")

    def _on_label_click(self, point_id, row_data, label_type, tick_button, cross_button):
        """Handle tick/cross button click for labeling tiles."""
        if point_id not in self.cached_embeddings:
            self._fetch_embeddings([point_id])

        is_positive = label_type == "pos"
        was_selected = (is_positive and point_id in self.pos_ids) or (not is_positive and point_id in self.neg_ids)
        
        if point_id in self.pos_ids: self.pos_ids.remove(point_id)
        if point_id in self.neg_ids: self.neg_ids.remove(point_id)
        
        if not was_selected:
            if is_positive:
                self.pos_ids.append(point_id)
                tick_button.button_style = "primary"
                tick_button.layout.opacity = "1.0"
                cross_button.button_style = ""
                cross_button.layout.opacity = "0.3"
                self._show_operation_status(f"✅ Labeled tile as Positive")
            else:
                self.neg_ids.append(point_id)
                cross_button.button_style = "danger"
                cross_button.layout.opacity = "1.0"
                tick_button.button_style = ""
                tick_button.layout.opacity = "0.3"
                self._show_operation_status(f"✅ Labeled tile as Negative")
        else:
            tick_button.button_style = ""
            cross_button.button_style = ""
            tick_button.layout.opacity = "0.3"
            cross_button.layout.opacity = "0.3"
            self._show_operation_status(f"✅ Removed label from tile")
        
        self.update_layers()
        self.update_query_vector()

    def _on_center_map_click(self, geom):
        """Handle map button click to pan and zoom to tile location."""
        try:
            lat, lon = geom.y, geom.x
            self.map.center = (lat, lon)
            self.map.zoom = 14
            
            half_size = 0.0025 / 2
            square_coords = [
                (lon - half_size, lat - half_size), (lon + half_size, lat - half_size),
                (lon + half_size, lat + half_size), (lon - half_size, lat + half_size),
                (lon - half_size, lat - half_size)
            ]
            
            from shapely.geometry import Polygon
            square_poly = Polygon(square_coords)
            
            for layer in self.map.layers:
                if hasattr(layer, 'name') and layer.name == 'tile_highlight':
                    self.map.remove_layer(layer)
            
            highlight_layer = ipyl.GeoJSON(
                data={"type": "Feature", "geometry": shapely.geometry.mapping(square_poly)},
                name='tile_highlight', style={'color': 'yellow', 'fillOpacity': 0.1, 'weight': 3}
            )
            self.map.add_layer(highlight_layer)
        except Exception as e:
            if self.verbose:
                print(f"Error centering map: {e}")
