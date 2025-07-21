"""Interactive map interface for geospatial similarity search using satellite embeddings."""

import json
import warnings
from datetime import datetime
from typing import Dict, Optional

import duckdb
import ee
import geopandas as gpd
import ipyleaflet as ipyl
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

from .ee_tools import (
    get_s2_rgb_median,
    get_s2_ndvi_median,
    get_s2_ndwi_median,
    get_ee_image_url,
)
from .ui_config import (
    UIConstants,
    BasemapConfig,
    DatabaseConstants,
    LayerStyles,
)
from .ee_tools import initialize_ee_with_credentials
from .utils import get_database_centroid
from .models import Config

warnings.simplefilter("ignore", category=FutureWarning)

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn(
        "MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key."
    )


class GeoVibes:
    """Interactive map interface for geospatial similarity search using satellite embeddings.

    Provides point-and-click labeling interface with similarity search capabilities
    using vector embeddings stored in DuckDB with HNSW indexing.
    """

    @classmethod
    def create(
        cls,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
        gcp_project: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Create a GeoVibes instance with Config model.

        Args:
            config: Config instance with AOIs and options
            config_path: Path to YAML config file to load
            gcp_project: Google Cloud Project ID for Earth Engine authentication (overrides config)
            verbose: Enable detailed progress messages (overrides config)
            **kwargs: Additional arguments

        Returns:
            GeoVibes instance
        """
        if config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided")

        if config is None:
            from .models import load_config_from_yaml

            config = load_config_from_yaml(config_path)

        return cls(
            config=config,
            gcp_project=gcp_project,
            verbose=verbose,
            **kwargs,
        )

    def __init__(
        self,
        config: Config,
        gcp_project: Optional[str] = None,
        duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None,
        baselayer_url: Optional[str] = None,
        disable_ee: bool = False,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Initialize GeoVibes interface with Config model.

        Args:
            config: Config instance containing AOIs and options.
            gcp_project: Google Cloud Project ID for Earth Engine authentication (overrides config).
            duckdb_connection: Existing DuckDB connection to reuse.
            baselayer_url: Custom basemap tile URL.
            disable_ee: Disable Earth Engine basemaps.
            verbose: Enable detailed progress messages (overrides config).
            **kwargs: Additional arguments for backwards compatibility.

        Raises:
            ValueError: If no AOIs are provided in config.
            RuntimeError: If there is an error connecting to the database.
        """
        self.config = config

        # Use verbose from parameter or config
        self.verbose = verbose if verbose is not None else config.options.verbose

        if self.verbose:
            print("Initializing GeoVibes...")

        # Validate config
        if not config.aois:
            raise ValueError("At least one AOI must be provided in config")

        # Use gcp_project parameter or extract from environment/config
        self.gcp_project = gcp_project

        self.ee_available = not disable_ee and initialize_ee_with_credentials(
            self.gcp_project
        )

        # Initialize AOI state
        self.current_aoi = config.aois[0]  # Start with first AOI
        self.available_databases = list(
            self.current_aoi.dbs.items()
        )  # List of (name, path) tuples
        self.current_database_name = (
            list(self.current_aoi.dbs.keys())[0] if self.current_aoi.dbs else None
        )
        self.current_database_path = (
            list(self.current_aoi.dbs.values())[0] if self.current_aoi.dbs else None
        )

        if self.verbose:
            print(f"📍 Starting with AOI: {self.current_aoi.name}")
            if self.current_database_name:
                print(f"💾 Starting with database: {self.current_database_name}")

        if baselayer_url is None:
            # Use AOI-specific basemap if available, otherwise default
            if self.current_aoi.basemap_tile_server_url:
                baselayer_url = self.current_aoi.basemap_tile_server_url
            else:
                baselayer_url = BasemapConfig.BASEMAP_TILES["MAPTILER"]

        # Set up database connection
        if duckdb_connection is None:
            if self.current_database_path is None:
                raise ValueError("No database available for the current AOI")

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

            # Configure memory limits to prevent kernel crashes
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False

        self.current_basemap = "MAPTILER"
        self.basemap_layer = ipyl.TileLayer(
            url=baselayer_url,
            no_wrap=True,
            name="basemap",
            attribution=BasemapConfig.MAPTILER_ATTRIBUTION,
        )

        # Set up Earth Engine boundary for current AOI
        if self.ee_available:
            try:
                self.ee_boundary = ee.Geometry(
                    shapely.geometry.mapping(
                        gpd.read_file(self.current_aoi.boundary).union_all()
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
            extension_queries = DatabaseConstants.get_extension_setup_queries(
                self.current_database_path
            )
            for query in extension_queries:
                try:
                    self.duckdb_connection.execute(query)
                    if self.verbose and "httpfs" in query:
                        print("📦 httpfs extension loaded for GCS support")
                    elif self.verbose and "spatial" in query:
                        print("🗺️  spatial extension loaded for geometry support")
                except Exception as e:
                    if "httpfs" in query:
                        raise RuntimeError(
                            f"Failed to load httpfs extension for GCS support: {str(e)}"
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to load required extension: {str(e)}"
                        )

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

        # Build UI
        self.side_panel, self.ui_widgets = self._build_side_panel()

        # Build results panel
        self.results_panel, self.results_widgets = self._build_results_panel()

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
            [self.side_panel, map_with_overlays, self.results_panel],
            layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width="100%"),
        )

        display(self.main_layout)

    def _setup_ee_basemaps(self) -> None:
        """Set up Earth Engine basemaps (Sentinel-2 RGB, NDVI, NDWI) if available."""
        self.basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()

        if self.ee_available and self.ee_boundary is not None:
            try:
                if self.verbose:
                    print("🛰️ Setting up Earth Engine basemaps (S2 RGB, NDVI, NDWI)...")

                # Use dates from config.options
                start_date = self.config.options.ee_basemap_start_date.strftime(
                    "%Y-%m-%d"
                )
                end_date = self.config.options.ee_basemap_end_date.strftime("%Y-%m-%d")

                s2_rgb_median = get_s2_rgb_median(
                    self.ee_boundary, start_date, end_date
                )
                s2_rgb_url = get_ee_image_url(
                    s2_rgb_median, BasemapConfig.S2_RGB_VIS_PARAMS
                )
                self.basemap_tiles["S2_RGB"] = s2_rgb_url

                ndvi_median = get_s2_ndvi_median(self.ee_boundary, start_date, end_date)
                ndvi_url = get_ee_image_url(ndvi_median, BasemapConfig.NDVI_VIS_PARAMS)
                self.basemap_tiles["NDVI"] = ndvi_url

                ndwi_median = get_s2_ndwi_median(self.ee_boundary, start_date, end_date)
                ndwi_url = get_ee_image_url(ndwi_median, BasemapConfig.NDWI_VIS_PARAMS)
                self.basemap_tiles["NDWI"] = ndwi_url

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
        )
        return map_widget

    def _build_side_panel(self):
        """Build the collapsible side panel with accordion sections."""
        self.search_btn = Button(
            description="Search",
            layout=Layout(width="100%", height=UIConstants.BUTTON_HEIGHT),
            button_style="success",  # Green to highlight importance
            tooltip="Find points similar to your positive labels",
        )

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
            [self.search_btn, self.neighbors_slider, self.reset_btn],
            layout=Layout(padding="5px", margin="0 0 10px 0"),
        )

        # --- AOI Selection section ---
        self.aoi_dropdown = None
        aoi_section_widgets = []
        if len(self.config.aois) > 1:
            # Create dropdown with AOI names
            aoi_options = [(aoi.name, aoi) for aoi in self.config.aois]
            self.aoi_dropdown = Dropdown(
                options=aoi_options,
                value=self.current_aoi,
                description="",
                layout=Layout(width="100%"),
            )
            aoi_section_widgets.append(Label("Select AOI:"))
            aoi_section_widgets.append(self.aoi_dropdown)

        # --- Database Selection section ---
        self.database_dropdown = None
        database_section_widgets = []
        if self.current_aoi.dbs:
            # Create dropdown with database names for current AOI
            database_options = [
                (name, path) for name, path in self.current_aoi.dbs.items()
            ]
            self.database_dropdown = Dropdown(
                options=database_options,
                value=self.current_database_path,
                description="",
                layout=Layout(width="100%"),
            )
            database_section_widgets.append(Label("Select Database:"))
            database_section_widgets.append(self.database_dropdown)

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

        # Use instance basemap_tiles which includes EE basemaps (NDVI/NDWI)
        basemap_tiles_to_use = getattr(
            self, "basemap_tiles", BasemapConfig.BASEMAP_TILES
        )

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

        # Build accordion - conditionally include AOI and database sections
        accordion_children = []
        accordion_titles = []

        # Add AOI section if multiple AOIs available
        if aoi_section_widgets:
            accordion_children.append(
                VBox(aoi_section_widgets, layout=Layout(padding="5px"))
            )
            accordion_titles.append("Area of Interest")

        # Add database section if databases available
        if database_section_widgets:
            accordion_children.append(
                VBox(database_section_widgets, layout=Layout(padding="5px"))
            )
            accordion_titles.append("Database")

        # Add other sections
        accordion_children.extend(
            [
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
                    ],
                    layout=Layout(padding="5px"),
                ),
            ]
        )

        accordion_titles.extend(["Label Mode", "Basemaps", "Export & Tools"])

        accordion = Accordion(children=accordion_children)

        # Set titles
        for i, title in enumerate(accordion_titles):
            accordion.set_title(i, title)

        # Open label mode by default
        label_model_index = accordion_titles.index("Label Mode")
        accordion.selected_index = label_model_index

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
            "aoi_dropdown": self.aoi_dropdown,
            "database_dropdown": self.database_dropdown,
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

    def _build_results_panel(self):
        """Build the collapsible results panel showing similar point chips."""

        # Results panel collapse/expand button
        self.results_collapse_btn = Button(
            description="▶",
            layout=Layout(
                width="30px",
                height="30px",
            ),
            tooltip="Show/Hide Results Panel",
        )

        # Panel header
        results_header = HBox(
            [
                Label("Similar Points", layout=Layout(flex="1")),
                self.results_collapse_btn,
            ],
            layout=Layout(width="100%", justify_content="space-between", padding="2px"),
        )

        # Results container that will hold the chips
        self.results_container = VBox(
            [],
            layout=Layout(
                width="100%",
                height="100%",  # Account for header height
                overflow_y="auto",
                padding="5px",
            ),
        )

        # Results content (header + container)
        self.results_content = VBox(
            [self.results_container], layout=Layout(width="100%", height="100%")
        )

        # Main results panel (initially collapsed)
        self.results_panel_collapsed = True
        panel_content = VBox(
            [
                results_header,
                self.results_content,
            ],
            layout=Layout(
                width="0px",  # Start collapsed
                height="100%",  # Match main layout height
                padding="5px",
                border="1px solid #ccc",
                display="none",  # Start hidden
            ),
        )

        # Wire the collapse button
        self.results_collapse_btn.on_click(self._on_toggle_results_collapse)

        # Return panel and widget references
        results_widgets = {
            "results_collapse_btn": self.results_collapse_btn,
            "results_container": self.results_container,
        }

        return panel_content, results_widgets

    def _on_toggle_results_collapse(self, b):
        """Toggle results panel collapse/expand."""
        if self.results_panel_collapsed:
            # Expand
            self.results_panel.layout.display = "flex"
            self.results_panel.layout.width = "250px"
            self.results_collapse_btn.description = "◀"
            self.results_panel_collapsed = False
        else:
            # Collapse
            self.results_panel.layout.display = "none"
            self.results_panel.layout.width = "0px"
            self.results_collapse_btn.description = "▶"
            self.results_panel_collapsed = True

    def _update_results_panel(self, search_results_df):
        """Update the results panel with chips for each similar point.

        Args:
            search_results_df: DataFrame with columns ['id', 'geometry_wkt', 'distance']
        """
        # Clear existing chips
        self.results_container.children = []

        if search_results_df.empty:
            return

        chips = []

        # Create a chip for each result
        for idx, row in search_results_df.head(14).iterrows():  # Limit to top 10
            try:
                # Extract lat/lon from geometry
                geom = shapely.wkt.loads(row["geometry_wkt"])
                lat, lon = geom.y, geom.x

                # Create placeholder image
                img_data = self._create_placeholder_png(lat, lon)

                # Create image widget
                img_widget = HTML(
                    value=f'<img src="{img_data}" width="64" height="64" style="border-radius: 4px; display: block; margin: 0 auto;">',
                    layout=Layout(width="100%", text_align="center"),
                )

                # Create distance text below image
                distance_text = HTML(
                    value=f"""
                    <div style="font-size: 9px; text-align: center; line-height: 1.1;">
                        <div><strong>Dist:</strong> {row["distance"]:.3f}</div>
                    </div>
                    """,
                    layout=Layout(width="100%"),
                )

                # Create chip container with vertical layout
                chip = VBox(
                    [img_widget, distance_text],
                    layout=Layout(
                        width="110px",  # Narrower for 2-column layout
                        height="90px",  # Taller to accommodate vertical layout
                        margin="2px",
                        border="1px solid #ddd",
                        border_radius="6px",
                        background_color="#f9f9f9",
                        padding="4px",
                        align_items="center",
                    ),
                )

                chips.append(chip)

            except Exception as e:
                if self.verbose:
                    print(f"Error creating chip for result {idx}: {e}")
                continue

        # Arrange chips in 2-column layout
        rows = []
        for i in range(0, len(chips), 2):
            if i + 1 < len(chips):
                # Two chips in this row
                row = HBox(
                    [chips[i], chips[i + 1]],
                    layout=Layout(
                        width="100%", justify_content="space-between", margin="1px 0"
                    ),
                )
            else:
                # Single chip in last row
                row = HBox(
                    [chips[i]],
                    layout=Layout(
                        width="100%", justify_content="flex-start", margin="1px 0"
                    ),
                )
            rows.append(row)

        # Update the container with new rows
        self.results_container.children = rows

        # Auto-expand results panel if it's collapsed and we have results
        if self.results_panel_collapsed and chips:
            self._on_toggle_results_collapse(None)

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

        # Label toggle
        self.label_toggle.observe(self._on_label_change, "value")

        # Selection mode toggle
        self.selection_mode.observe(self._on_selection_mode_change, "value")

        # # Neighbors slider
        # self.neighbors_slider.observe(self._on_neighbors_change, 'value')

        # Basemap buttons
        for basemap_name, btn in self.basemap_buttons.items():
            btn.on_click(lambda b, name=basemap_name: self._on_basemap_select(name))

        # AOI dropdown
        if self.aoi_dropdown:
            self.aoi_dropdown.observe(self._on_aoi_change, names=["value"])

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
        lat, lon = kwargs.get("coordinates", (0, 0))

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
            return

        # Normal label point behavior
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

        # Clear all label lists
        self.pos_ids = []
        self.neg_ids = []

        # Clear cached embeddings
        self.cached_embeddings = {}

        # Reset query vector
        self.query_vector = None

        # Clear detections
        self.detections_with_embeddings = None

        # Clear all map layers
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

        # Clear results panel
        self.results_container.children = []

        # Clear operation status
        self._clear_operation_status()

        if self.verbose:
            print("✅ All data cleared!")

    def _fetch_embeddings(self, point_ids, chunk_size=None):
        """Fetch embeddings for given point IDs in chunks and cache them."""
        if chunk_size is None:
            chunk_size = DatabaseConstants.EMBEDDING_CHUNK_SIZE

        missing_ids = [pid for pid in point_ids if pid not in self.cached_embeddings]

        if not missing_ids:
            return

        # Show progress for large batches
        if len(missing_ids) > 100:
            self._show_operation_status(
                f"🔄 Fetching embeddings for {len(missing_ids)} points..."
            )
            if self.verbose:
                print(f"🔄 Fetching embeddings for {len(missing_ids)} points...")

        # Process in chunks to avoid memory issues
        for i in range(0, len(missing_ids), chunk_size):
            chunk = missing_ids[i : i + chunk_size]

            # Show chunk progress for very large batches
            if len(missing_ids) > chunk_size:
                chunk_num = i // chunk_size + 1
                total_chunks = (len(missing_ids) - 1) // chunk_size + 1
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
            SELECT id, embedding
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """

            # Fetch as Arrow then convert to pandas
            arrow_table = self.duckdb_connection.execute(
                query, prepared_chunk
            ).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()

            # Cache the embeddings from this chunk
            for _, row in chunk_df.iterrows():
                embedding = np.array(row["embedding"])
                # Ensure consistent string type for point IDs
                point_id = str(row["id"])
                self.cached_embeddings[point_id] = embedding

        if len(missing_ids) > 100:
            self._show_operation_status(
                f"✅ Cached embeddings for {len(missing_ids)} points"
            )
            if self.verbose:
                print(f"✅ Cached embeddings for {len(missing_ids)} points")

    def search_click(self, b):
        """Perform similarity search based on current query vector."""
        if self.query_vector is None:
            if self.verbose:
                print(
                    "⚠️ No query vector available. Please add some positive labels first."
                )
            return

        n_neighbors = self.neighbors_slider.value

        # Convert query vector to the format needed for DuckDB
        query_vec = self.query_vector.tolist()

        # Get labeled IDs for post-filtering
        all_labeled_ids = self.pos_ids + self.neg_ids

        # Request extra results to account for filtering out labeled points
        # Add some buffer (max 50% extra) to ensure we get enough results after filtering
        extra_results = min(len(all_labeled_ids), n_neighbors // 2)
        total_requested = n_neighbors + extra_results

        # Use dynamic query with detected embedding dimension
        sql = DatabaseConstants.get_similarity_search_light_query(self.embedding_dim)
        query_params = [query_vec, total_requested]

        # Show search progress in status bar
        if all_labeled_ids:
            self._show_operation_status(
                f"🔍 Searching for {n_neighbors} points (will filter {len(all_labeled_ids)} labeled)..."
            )
            if self.verbose:
                print(
                    f"🔍 Searching for {n_neighbors} similar points (requesting {total_requested}, will filter {len(all_labeled_ids)} labeled points)..."
                )
        else:
            self._show_operation_status(
                f"🔍 Searching for {n_neighbors} similar points..."
            )
            if self.verbose:
                print(f"🔍 Searching for {n_neighbors} similar points...")

        # Fetch as Arrow table then convert only needed columns to pandas
        arrow_table = self.duckdb_connection.execute(
            sql, query_params
        ).fetch_arrow_table()
        search_results = arrow_table.select(
            ["id", "geometry_json", "geometry_wkt", "distance"]
        ).to_pandas()

        # Post-filter to exclude labeled points in memory (much safer than DuckDB NOT IN)
        if all_labeled_ids:
            # Convert to string IDs for consistent comparison
            labeled_id_strings = set(str(lid) for lid in all_labeled_ids)
            # Filter out labeled points
            mask = ~search_results["id"].astype(str).isin(labeled_id_strings)
            search_results_filtered = search_results[mask].copy()
            # Take only the requested number of neighbors
            search_results_filtered = search_results_filtered.head(n_neighbors)
        else:
            search_results_filtered = search_results.head(n_neighbors)

        # Show results in status bar
        filtered_count = len(search_results_filtered)
        if all_labeled_ids:
            total_found = len(search_results)
            filtered_out = total_found - filtered_count
            self._show_operation_status(
                f"✅ Found {filtered_count} similar points (filtered out {filtered_out} labeled)"
            )
            if self.verbose:
                print(
                    f"✅ Found {filtered_count} similar points after filtering out {filtered_out} labeled points"
                )
        else:
            self._show_operation_status(f"✅ Found {filtered_count} similar points")

        # Create geometries from WKT
        geometries = [
            shapely.wkt.loads(row["geometry_wkt"]) if row["geometry_wkt"] else None
            for _, row in search_results_filtered.iterrows()
        ]

        # Create detections DataFrame without embeddings (fetched on-demand during labeling)
        self.detections_with_embeddings = gpd.GeoDataFrame(
            {
                "id": search_results_filtered["id"]
                .astype(str)
                .values,  # Ensure string type
                "distance": search_results_filtered["distance"].values,
                "geometry": geometries,
            }
        )

        # Create GeoJSON for map display with distance-based coloring
        detections_geojson = {"type": "FeatureCollection", "features": []}

        if not search_results_filtered.empty:
            # Calculate distance range for color mapping
            min_distance = search_results_filtered["distance"].min()
            max_distance = search_results_filtered["distance"].max()

            for _, row in search_results_filtered.iterrows():
                # Calculate color based on distance
                color = UIConstants.distance_to_color(
                    row["distance"], min_distance, max_distance
                )

                detections_geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(row["geometry_json"]),
                        "properties": {
                            "id": str(row["id"]),
                            "distance": row["distance"],
                            "color": color,
                            "fillColor": color,
                        },
                    }
                )

        # Update the map with distance-colored points
        self._update_search_layer_with_colors(detections_geojson)

        # Update the results panel with similar point chips
        self._update_results_panel(search_results_filtered)

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        # Don't process clicks when in polygon mode or actively drawing
        if not self.execute_label_point or self.lasso_mode or self.polygon_drawing:
            return

        action = kwargs.get("type")
        if action not in ["click"]:
            return

        lat, lon = kwargs.get("coordinates")

        clicked_point = Point(lon, lat)
        point_id = None

        # First check if we have cached detections
        if (
            self.detections_with_embeddings is not None
            and len(self.detections_with_embeddings) > 0
        ):
            # Find nearest point in cached detections
            distances = self.detections_with_embeddings.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()

            # Use a threshold to ensure we're clicking on an actual point
            if distances[nearest_idx] < UIConstants.CLICK_THRESHOLD:
                nearest_detection = self.detections_with_embeddings.loc[nearest_idx]
                point_id = str(nearest_detection["id"])  # Ensure string type

        # If not found in cache, query the database
        if point_id is None:
            # Use lightweight query without embedding
            sql = DatabaseConstants.NEAREST_POINT_LIGHT_QUERY

            arrow_table = self.duckdb_connection.execute(
                sql, [lon, lat]
            ).fetch_arrow_table()
            nearest_result = arrow_table.to_pandas()

            if nearest_result.empty:
                return

            point_id = str(nearest_result.iloc[0]["id"])  # Convert to string

        # Fetch embedding on-demand for this specific point
        self._fetch_embeddings([point_id])

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
        self.update_query_vector()  # Don't skip fetch to ensure query vector is properly computed

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

    def _update_ee_boundary(self):
        """Update Earth Engine boundary based on current AOI boundary."""
        if not self.ee_available:
            return

        if self.current_aoi.boundary:
            try:
                boundary_gdf = gpd.read_file(self.current_aoi.boundary)
                self.ee_boundary = ee.Geometry(
                    shapely.geometry.mapping(boundary_gdf.union_all())
                )
                if self.verbose:
                    print(
                        f"🛰️ Updated Earth Engine boundary from: {self.current_aoi.boundary}"
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

        # Add new boundary layer if we have a boundary in current AOI
        if self.current_aoi.boundary:
            try:
                if self.verbose:
                    print(f"🗺️  Loading boundary layer: {self.current_aoi.boundary}")

                # Use geopandas to read the file (handles both local and GCS paths)
                boundary_gdf = gpd.read_file(self.current_aoi.boundary)

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
        # Use boundary from current AOI
        boundary_path = self.current_aoi.boundary

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

        # Use database centroid for centering if boundary failed or not available
        if (
            self.current_database_path
            and hasattr(self, "duckdb_connection")
            and self.duckdb_connection
        ):
            center_y, center_x = get_database_centroid(
                self.duckdb_connection, verbose=self.verbose
            )
        else:
            # Fallback to a default center if no database connection
            center_y, center_x = 0.0, 0.0
            if self.verbose:
                print(
                    "⚠️  No database connection available, using default center (0, 0)"
                )

        return center_y, center_x

    def _warm_up_gcs_database(self):
        """Warm up GCS database with initial search for better performance."""
        try:
            if self.verbose:
                print("🔧 Optimizing database connection...")

            # Get the first point's embedding from the database
            first_point_query = """
            SELECT embedding 
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
        """Handle database selection change within current AOI."""
        new_database_path = change["new"]

        if new_database_path == self.current_database_path:
            return  # No change

        # Find the database name from the current AOI
        new_database_name = None
        for name, path in self.current_aoi.dbs.items():
            if path == new_database_path:
                new_database_name = name
                break

        if self.verbose:
            print(f"🔄 Switching to database: {new_database_name or 'Unknown'}")

        # Show loading message immediately
        self._show_operation_status(
            "🔄 Loading database (this can take a couple of seconds)..."
        )

        try:
            # Update database references
            old_database_path = self.current_database_path
            self.current_database_path = new_database_path
            self.current_database_name = new_database_name

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

            # Configure memory limits
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)

            # Setup extensions
            extension_queries = DatabaseConstants.get_extension_setup_queries(
                new_database_path
            )
            for query in extension_queries:
                self.duckdb_connection.execute(query)

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
                f"✅ Successfully loaded: {new_database_name or 'database'}"
            )
            if self.verbose:
                print(f"✅ Successfully switched to database: {new_database_name}")

        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch database: {str(e)}")

            # Revert to previous database
            self.current_database_path = old_database_path
            if self.database_dropdown:
                self.database_dropdown.value = old_database_path

            self._show_operation_status(f"❌ Failed to load database: {str(e)}")

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

        # Clear results panel
        self.results_container.children = []

        # Clear operation status
        self._clear_operation_status()

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_owns_connection") and self._owns_connection:
            if hasattr(self, "duckdb_connection") and self.duckdb_connection:
                self.duckdb_connection.close()
                if self.verbose:
                    print("🔌 DuckDB connection closed.")

    def _on_aoi_change(self, change):
        """Handle AOI selection change."""
        new_aoi = change["new"]

        if new_aoi == self.current_aoi:
            return  # No change

        if self.verbose:
            print(f"🔄 Switching to AOI: {new_aoi.name}")

        # Show loading message immediately
        self._show_operation_status(
            "🔄 Loading AOI (this can take a couple of seconds)..."
        )

        try:
            # Step 1: Quick UI updates - update AOI and boundary first
            old_aoi = self.current_aoi
            self.current_aoi = new_aoi

            # Update available databases for the new AOI
            self.available_databases = list(self.current_aoi.dbs.items())

            # Set the first database as current if available
            if self.current_aoi.dbs:
                self.current_database_name = list(self.current_aoi.dbs.keys())[0]
                self.current_database_path = list(self.current_aoi.dbs.values())[0]
            else:
                self.current_database_name = None
                self.current_database_path = None

            # Update database dropdown options
            if self.database_dropdown:
                if self.current_aoi.dbs:
                    database_options = [
                        (name, path) for name, path in self.current_aoi.dbs.items()
                    ]
                    self.database_dropdown.options = database_options
                    self.database_dropdown.value = self.current_database_path
                else:
                    self.database_dropdown.options = []

            # Update boundary path and recenter map
            lat, lon = self._setup_boundary_and_center()
            self.map.center = (lat, lon)

            if self.verbose:
                print(f"📍 Map recentered to: {lat:.4f}, {lon:.4f}")

            # Update Earth Engine boundary
            self._update_ee_boundary()

            # Update boundary layer on map
            self._update_boundary_layer()

            # Step 2: Heavy database operations if we have a database
            if self.current_database_path:
                self._show_operation_status("🔄 Connecting to new database...")

                # Close current connection if we own it
                if hasattr(self, "_owns_connection") and self._owns_connection:
                    if hasattr(self, "duckdb_connection") and self.duckdb_connection:
                        self.duckdb_connection.close()

                # Reset all application state
                self._reset_all_state()

                # Establish new connection
                self.duckdb_connection = DatabaseConstants.setup_duckdb_connection(
                    self.current_database_path, read_only=True
                )
                self._owns_connection = True

                # Configure memory limits
                for query in DatabaseConstants.get_memory_setup_queries():
                    self.duckdb_connection.execute(query)

                # Setup extensions
                extension_queries = DatabaseConstants.get_extension_setup_queries(
                    self.current_database_path
                )
                for query in extension_queries:
                    self.duckdb_connection.execute(query)

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
                if DatabaseConstants.is_gcs_path(self.current_database_path):
                    self._show_operation_status("🔄 Optimizing database connection...")
                    self._warm_up_gcs_database()

            # Update Earth Engine basemaps for new AOI
            if self.ee_available and self.ee_boundary is not None:
                self._show_operation_status("🔄 Updating Earth Engine basemaps...")
                self._setup_ee_basemaps()

            # Success!
            self._show_operation_status(f"✅ Successfully loaded AOI: {new_aoi.name}")
            if self.verbose:
                print(f"✅ Successfully switched to AOI: {new_aoi.name}")
                if self.current_database_name:
                    print(f"✅ Using database: {self.current_database_name}")

        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch AOI: {str(e)}")

            # Revert to previous AOI
            self.current_aoi = old_aoi
            if self.aoi_dropdown:
                self.aoi_dropdown.value = old_aoi

            self._show_operation_status(f"❌ Failed to load AOI: {str(e)}")

            # Try to restore previous map position
            try:
                lat, lon = self._setup_boundary_and_center()
                self.map.center = (lat, lon)
            except:
                pass  # If this fails too, just leave the map where it is
