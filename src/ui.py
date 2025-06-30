"""Interactive map interface for geospatial similarity search using satellite embeddings."""

import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import duckdb
import ee
import geopandas as gpd
import ipyleaflet as ipyl
from ipyleaflet import Map, DrawControl, Heatmap
from IPython.display import display
from ipywidgets import Button, VBox, HBox, IntSlider, Label, Layout, HTML, ToggleButtons, Accordion, FileUpload, Output, Checkbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
import webbrowser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from .ee_tools import get_s2_rgb_median, get_s2_ndvi_median, get_s2_ndwi_median, get_ee_image_url, initialize_ee_with_credentials
from .ui_config import UIConstants, BasemapConfig, GeoVibesConfig, DatabaseConstants, LayerStyles

warnings.simplefilter("ignore", category=FutureWarning)

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn("MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key.")


class GeoVibes:
    """Interactive map interface for geospatial similarity search using satellite embeddings.
    
    Provides point-and-click labeling interface with similarity search capabilities
    using vector embeddings stored in DuckDB with HNSW indexing.
    """
    
    @classmethod
    def from_config(cls, config_path, verbose=False, **kwargs):
        """Create a GeoLabeler instance from a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            verbose: If True, print detailed progress messages
            **kwargs: Additional keyword arguments to override config values
        
        Returns:
            GeoLabeler instance
        """
        return cls(config_path=config_path, verbose=verbose, **kwargs)
    def __init__(
            self, 
            geojson_path: Optional[str] = None, 
            start_date: Optional[str] = None, 
            end_date: Optional[str] = None,
            duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None, 
            duckdb_path: Optional[str] = None, 
            config: Optional[Dict] = None, 
            config_path: Optional[str] = None,
            baselayer_url: Optional[str] = None, 
            verbose: bool = False, 
            **kwargs) -> None:
        """Initialize GeoVibes interface.
        
        Args:
            geojson_path: Path to boundary GeoJSON file.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            duckdb_connection: Existing DuckDB connection to reuse.
            duckdb_path: Path to DuckDB database file.
            config: Configuration dictionary.
            config_path: Path to JSON configuration file.
            baselayer_url: Custom basemap tile URL.
            verbose: Enable detailed progress messages.
        """
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes...")
        
        if config_path is not None:
            self.config = GeoVibesConfig.from_file(config_path)
            self.config.validate()
        elif config is not None:
            self.config = GeoVibesConfig.from_dict(config)
            self.config.validate()
        else:
            if geojson_path is None or start_date is None or end_date is None:
                raise ValueError("Required parameters missing. Provide either config_path, config dict, or individual parameters.")
            self.config = GeoVibesConfig(
                duckdb_path=duckdb_path,
                boundary_path=geojson_path,
                start_date=start_date,
                end_date=end_date
            )
            self.config.validate()
        
        self.ee_available = initialize_ee_with_credentials(self.config.gcp_project)
        
        if baselayer_url is None:
            baselayer_url = BasemapConfig.BASEMAP_TILES['MAPTILER']
        
        if duckdb_connection is None:
            self.duckdb_connection = duckdb.connect(self.config.duckdb_path, read_only=True)
            self._owns_connection = True
            
            # Configure memory limits to prevent kernel crashes
            for query in DatabaseConstants.get_memory_setup_queries():
                self.duckdb_connection.execute(query)
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False
        self.current_basemap = 'MAPTILER'
        self.basemap_layer = ipyl.TileLayer(url=baselayer_url, no_wrap=True, name='basemap', 
                                       attribution=BasemapConfig.MAPTILER_ATTRIBUTION)
        if self.ee_available:
            try:
                self.ee_boundary = ee.Geometry(shapely.geometry.mapping(
                    gpd.read_file(self.config.boundary_path).union_all()))
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to create Earth Engine boundary: {e}")
                    print("⚠️  NDVI/NDWI basemaps will be unavailable")
                self.ee_boundary = None
        else:
            self.ee_boundary = None
        
        # Setup spatial extension in DuckDB
        self.duckdb_connection.execute(DatabaseConstants.EXTENSION_SETUP_QUERY)

        # Detect embedding dimension from database
        try:
            self.embedding_dim = DatabaseConstants.detect_embedding_dimension(self.duckdb_connection)
            if self.verbose:
                print(f"🔍 Detected embedding dimension: {self.embedding_dim}")
        except ValueError as e:
            if self.verbose:
                print(f"⚠️ Could not detect embedding dimension: {e}")
                print("⚠️ Using default dimension of 1000")
            self.embedding_dim = 384

        # Get centroid of boundary from geopandas
        boundary_gdf = gpd.read_file(self.config.boundary_path)
        center_y, center_x = boundary_gdf.geometry.iloc[0].centroid.y, boundary_gdf.geometry.iloc[0].centroid.x
        
        # Build map
        self.map = self._build_map(center_y, center_x)

        # Add Earth Engine basemap options (if available)
        self._setup_ee_basemaps()

        if self.verbose:
            print("Building UI...")
        
        # Initialize state
        self.current_label = 'Positive'
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
        
        # Linear classifier state
        self.classifier = None
        self.scaler = None
        self.predictions_gdf = None
        self.classifier_trained = False
        self.show_heatmap = False
        self.normalize_colors = True
        self.highlight_training_points = True

        
        # Build UI
        self.side_panel, self.ui_widgets = self._build_side_panel()
        self.heatmap_toggle.value = False
        self.normalize_heatmap.value = True
        self.highlight_training.value = True
        # Add layers to map
        self._add_map_layers()
        
        # Add DrawControl
        self._setup_draw_control()
        
        # Wire events
        self._wire_events()
        
        # Add legend (will be updated dynamically based on coloring mode)
        self._update_legend()
        
        # Add status bar
        self.status_bar = HTML(value="Ready")
        
        # Create main layout
        map_with_overlays = VBox([
            self.map,
            HBox([self.legend, self.status_bar], 
                 layout=Layout(justify_content='space-between', padding='5px'))
        ], layout=Layout(flex='1 1 auto'))
        
        self.main_layout = HBox([
            self.side_panel,
            map_with_overlays
        ], layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width='100%'))
        
        display(self.main_layout)


    def _setup_ee_basemaps(self) -> None:
        """Set up Earth Engine basemaps (Sentinel-2 RGB, NDVI, NDWI) if available."""
        self.basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()
        
        if self.ee_available and self.ee_boundary is not None:
            try:
                if self.verbose:
                    print("🛰️ Setting up Earth Engine basemaps (S2 RGB, NDVI, NDWI)...")
                
                s2_rgb_median = get_s2_rgb_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date)
                s2_rgb_url = get_ee_image_url(s2_rgb_median, BasemapConfig.S2_RGB_VIS_PARAMS)
                self.basemap_tiles['S2_RGB'] = s2_rgb_url
                
                ndvi_median = get_s2_ndvi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date)
                ndvi_url = get_ee_image_url(ndvi_median, BasemapConfig.NDVI_VIS_PARAMS)
                self.basemap_tiles['NDVI'] = ndvi_url

                ndwi_median = get_s2_ndwi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date)
                ndwi_url = get_ee_image_url(ndwi_median, BasemapConfig.NDWI_VIS_PARAMS)
                self.basemap_tiles['NDWI'] = ndwi_url
                
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
            layout=Layout(flex='1 1 auto', height='100%'),
            scroll_wheel_zoom=True
        )
        return map_widget


    def _build_side_panel(self):
        """Build the collapsible side panel with accordion sections."""
        self.search_btn = Button(
            description='Search',
            layout=Layout(width='100%', height=UIConstants.BUTTON_HEIGHT),
            button_style='success',  # Green to highlight importance
            tooltip='Find points similar to your positive labels'
        )
        
        self.neighbors_slider = IntSlider(
            value=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            description='',  # No description
            readout=True,
            layout=Layout(width='100%')
        )
        
        self.histogram_output = Output(layout=Layout(width='100%', height='150px', display='none'))
        
        self.reset_btn = Button(
            description='🗑️ Reset',
            layout=Layout(width='100%', height=UIConstants.RESET_BUTTON_HEIGHT),
            button_style='',
            tooltip='Clear all labels and search results'
        )
        
        # --- Linear Classifier section ---
        self.classify_btn = Button(
            description='🤖 Classify All',
            layout=Layout(width='100%', height=UIConstants.BUTTON_HEIGHT),
            button_style='warning',  # Orange to distinguish from search
            tooltip='Train linear classifier on labels and predict all points'
        )
        
        self.heatmap_toggle = Checkbox(
            value=False,
            description='Show Heatmap',
            layout=Layout(width='100%'),
            tooltip='Toggle confidence heatmap layer visibility'
        )
        
        self.normalize_heatmap = Checkbox(
            value=True,
            description='Percentile Colors',
            layout=Layout(width='100%'),
            tooltip='Use percentile-based coloring (top 20% = darkest) vs absolute confidence values'
        )
        
        self.highlight_training = Checkbox(
            value=True,
            description='Highlight Training',
            layout=Layout(width='100%'),
            tooltip='Show training points with maximum confidence for visualization'
        )
        
        self.classifier_output = Output(layout=Layout(width='100%', height='150px', display='none'))
        
        search_section = VBox([
            self.search_btn,
            self.neighbors_slider,
            self.histogram_output,
            self.classify_btn,
            self.heatmap_toggle,
            self.normalize_heatmap,
            self.highlight_training,
            self.classifier_output,
            self.reset_btn
        ], layout=Layout(padding='5px', margin='0 0 10px 0'))
        
        # --- Labeling section ---
        self.label_toggle = ToggleButtons(
            options=[('Positive', 'Positive'), ('Negative', 'Negative'), ('Erase', 'Erase')],
            value='Positive',
            layout=Layout(width='100%')
        )
        
        # Add selection mode toggle
        self.selection_mode = ToggleButtons(
            options=[('Point', 'point'), ('Polygon', 'polygon')],
            value='point',
            layout=Layout(width='100%')
        )
        
        # Apply colors to toggle buttons
        self._update_toggle_button_styles()
        
        # --- Basemap Selection ---
        self.basemap_buttons = {}
        basemap_section_widgets = []
        
        # Use instance basemap_tiles which includes EE basemaps (NDVI/NDWI)
        basemap_tiles_to_use = getattr(self, 'basemap_tiles', BasemapConfig.BASEMAP_TILES)
        
        for basemap_name in basemap_tiles_to_use.keys():
            btn = Button(
                description=basemap_name.replace('_', ' '),
                layout=Layout(width='100%', margin='1px'),
                button_style=''
            )
            btn.basemap_name = basemap_name  # Store basemap name for reference
            self.basemap_buttons[basemap_name] = btn
            basemap_section_widgets.append(btn)
        
        # Highlight current basemap
        self._update_basemap_button_styles()
        
        # --- Export section ---
        self.save_btn = Button(description='💾 Save Dataset', layout=Layout(width='100%'))
        
        # --- Load Dataset section ---
        self.load_btn = Button(description='📂 Load Dataset', layout=Layout(width='100%'))
        self.file_upload = FileUpload(
            accept='.geojson,.parquet',
            multiple=False,
            layout=Layout(width='100%', display='none')  # Initially hidden
        )
        
        # --- External Tools section ---
        self.google_maps_btn = Button(
            description='🌍 Google Maps ↗',
            layout=Layout(width='100%'),
            button_style=''
        )
        
        # Build accordion
        accordion = Accordion(children=[
            VBox([
                Label('Label Type:'),
                self.label_toggle,
                Label('Selection Mode:', layout=Layout(margin='10px 0 0 0')),
                self.selection_mode
            ], layout=Layout(padding='5px')),
            VBox(basemap_section_widgets, layout=Layout(padding='5px')),
            VBox([self.save_btn, self.load_btn, self.file_upload, self.google_maps_btn], layout=Layout(padding='5px'))
        ])
        
        # Set titles
        for i, title in enumerate(['Label Mode', 'Basemaps', 'Export & Tools']):
            accordion.set_title(i, title)
        
        # Open label mode by default
        accordion.selected_index = 0
        
        # Add collapse/expand functionality
        self.panel_collapsed = False
        self.collapse_btn = Button(
            description='◀',
            layout=Layout(width=UIConstants.COLLAPSE_BUTTON_SIZE, height=UIConstants.COLLAPSE_BUTTON_SIZE),
            tooltip='Collapse/Expand Panel'
        )
        
        # Main panel with collapse button
        panel_header = HBox([
            Label('Controls', layout=Layout(flex='1')),
            self.collapse_btn
        ], layout=Layout(width='100%', justify_content='space-between', padding='2px'))
        
        # Create accordion container that will be hidden/shown
        self.accordion_container = VBox([accordion], layout=Layout(width='100%'))
        
        # Panel content includes search (always visible) and accordion (collapsible)
        panel_content = VBox([
            panel_header,
            search_section,  # Always visible
            self.accordion_container  # This will be hidden/shown
        ], layout=Layout(width=UIConstants.PANEL_WIDTH, padding='5px'))  # Narrower width
        
        # Return panel and widget references
        ui_widgets = {
            'search_btn': self.search_btn,
            'reset_btn': self.reset_btn,
            'label_toggle': self.label_toggle,
            'selection_mode': self.selection_mode,
            'neighbors_slider': self.neighbors_slider,
            'histogram_output': self.histogram_output,
            'classify_btn': self.classify_btn,
            'heatmap_toggle': self.heatmap_toggle,
            'normalize_heatmap': self.normalize_heatmap,
            'highlight_training': self.highlight_training,
            'classifier_output': self.classifier_output,
            'basemap_buttons': self.basemap_buttons,
            'save_btn': self.save_btn,
            'load_btn': self.load_btn,
            'file_upload': self.file_upload,
            'google_maps_btn': self.google_maps_btn,
            'collapse_btn': self.collapse_btn
        }
        
        return panel_content, ui_widgets


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
        # Region boundary
        with open(self.config.boundary_path) as f:
            region_layer = ipyl.GeoJSON(
                    name="region",
                    data=json.load(f),
                    style=LayerStyles.get_region_style()
                )
        self.map.add_layer(region_layer)

        # Positive layer
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.POS_COLOR)
        )
        self.map.add_layer(self.pos_layer)

        # Negative layer
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.NEG_COLOR)
        )
        self.map.add_layer(self.neg_layer)

        # Erase layer
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_erase_style()
        )
        self.map.add_layer(self.erase_layer)
        
        # Points layer for search results
        self.points = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_search_style(),
            hover_style=LayerStyles.get_search_hover_style()
        )
        self.map.add_layer(self.points)
        
        # Heatmap layer for classifier confidence scores
        self.heatmap_layer = Heatmap(
            locations=[],  # Empty initially, will be populated with [lat, lon, intensity] points
            radius=25,
            blur=10,
            max_intensity=1.0,
            gradient={0.0: 'navy', 0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        )
        self.map.add_layer(self.heatmap_layer)

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
        self.label_toggle.observe(self._on_label_change, 'value')
        
        # Selection mode toggle
        self.selection_mode.observe(self._on_selection_mode_change, 'value')
        
        # # Neighbors slider
        # self.neighbors_slider.observe(self._on_neighbors_change, 'value')
        
        # Basemap buttons
        for basemap_name, btn in self.basemap_buttons.items():
            btn.on_click(lambda b, name=basemap_name: self._on_basemap_select(name))
        
        # Collapse button
        self.collapse_btn.on_click(self._on_toggle_collapse)
        
        # Export and external tools
        self.save_btn.on_click(self.save_dataset)
        self.load_btn.on_click(self._on_load_click)
        self.file_upload.observe(self._on_file_upload, names=['value'])
        self.google_maps_btn.on_click(self._on_google_maps_click)
        
        # Classifier controls
        self.classify_btn.on_click(self.classify_all)
        self.heatmap_toggle.observe(self._on_heatmap_toggle, 'value')
        self.normalize_heatmap.observe(self._on_normalize_heatmap, 'value')
        self.highlight_training.observe(self._on_highlight_training, 'value')
        
        # Map interactions
        self.map.on_interaction(self._on_map_interaction)


    def _on_label_change(self, change):
        """Handle label toggle change."""
        self.current_label = change['new']
        if self.current_label == 'Positive':
            self.select_val = UIConstants.POSITIVE_LABEL
        elif self.current_label == 'Negative':
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
        if self.file_upload.layout.display == 'none':
            self.file_upload.layout.display = 'flex'
            self.load_btn.description = '📂 Cancel Load'
        else:
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'
            # Clear any uploaded files
            self.file_upload.value = ()

    def _on_file_upload(self, change):
        """Handle file upload."""
        if not change['new']:
            return
        
        # Get the uploaded file - change['new'] is a tuple of uploaded files
        uploaded_files = change['new']
        if not uploaded_files:
            return
            
        # Get the first uploaded file
        uploaded_file = uploaded_files[0]
        filename = uploaded_file['name']
        content = uploaded_file['content']
        
        try:
            self.load_dataset_from_content(content, filename)
            # Hide the upload widget and reset button text
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'
            # Clear the upload widget
            self.file_upload.value = ()
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            # Still hide the widget on error
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'


    def _on_basemap_select(self, basemap_name):
        """Handle basemap selection."""
        self.current_basemap = basemap_name
        # Use instance basemap_tiles which includes EE basemaps
        if hasattr(self, 'basemap_tiles'):
            self.basemap_layer.url = self.basemap_tiles[basemap_name]
        else:
            self.basemap_layer.url = BasemapConfig.BASEMAP_TILES[basemap_name]
        self._update_basemap_button_styles()


    def _on_toggle_collapse(self, b):
        """Toggle panel collapse/expand."""
        if self.panel_collapsed:
            # Expand
            self.accordion_container.layout.display = 'flex'
            self.collapse_btn.description = '◀'
            self.panel_collapsed = False
        else:
            # Collapse
            self.accordion_container.layout.display = 'none'
            self.collapse_btn.description = '▶'
            self.panel_collapsed = True


    def _on_map_interaction(self, **kwargs):
        """Handle all map interactions."""
        lat, lon = kwargs.get('coordinates', (0, 0))
        
        # Update status
        self._update_status(lat, lon)
        
        # Handle shift-click for polygon drawing hint
        if kwargs.get('type') == 'mousemove' and kwargs.get('modifiers', {}).get('shiftKey', False):
            self.status_bar.value += " | <b>Hold Shift + Draw to select multiple points</b>"
        
        # Handle ctrl-click for Google Maps
        if kwargs.get('type') == 'click' and kwargs.get('modifiers', {}).get('ctrlKey', False):
            url = f"https://www.google.com/maps/@{lat},{lon},18z"
            webbrowser.open(url, new=2)
            return
        
        # Normal label point behavior
        self.label_point(**kwargs)


    def _on_selection_mode_change(self, change):
        """Handle selection mode change."""
        self.lasso_mode = (change['new'] == 'polygon')
        self._update_status()

    def _on_heatmap_toggle(self, change):
        """Handle heatmap visibility toggle."""
        self.show_heatmap = change['new']
        if self.show_heatmap and self.predictions_gdf is not None:
            self._update_heatmap_layer()
        else:
            # Hide heatmap by clearing the locations
            self.heatmap_layer.locations = []

    def _on_normalize_heatmap(self, change):
        """Handle heatmap normalization toggle."""
        self.normalize_colors = change['new']
        self._update_legend()  # Update legend to reflect new mode
        if self.show_heatmap and self.predictions_gdf is not None:
            self._update_heatmap_layer()

    def _on_highlight_training(self, change):
        """Handle highlight training toggle."""
        self.highlight_training_points = change['new']
        if self.show_heatmap and self.predictions_gdf is not None:
            self._update_heatmap_layer()

    def handle_draw(self, target, action, geo_json):
        """Handle polygon drawing with chunked embedding fetching."""
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Mark that we're processing a polygon
            self.polygon_drawing = False
            
            # Get the polygon geometry from the drawn shape and convert to shapely Polygon
            polygon_coords = geo_json['geometry']['coordinates'][0]
            polygon = shapely.geometry.Polygon(polygon_coords)
            
            point_ids = []
            
            # First check cached detections
            if self.detections_with_embeddings is not None and len(self.detections_with_embeddings) > 0:
                # Find points within polygon from cached detections
                within_mask = self.detections_with_embeddings.geometry.within(polygon)
                cached_points = self.detections_with_embeddings[within_mask]
                
                point_ids.extend(cached_points['id'].tolist())
            
            # If no cached results or need more points, query the database
            if len(point_ids) == 0:
                polygon_wkt = polygon.wkt
                
                # Use lightweight query without embeddings
                points_in_polygon_query = f"""
                SELECT id
                FROM geo_embeddings
                WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
                """
                
                arrow_table = self.duckdb_connection.execute(points_in_polygon_query).fetch_arrow_table()
                points_inside = arrow_table.to_pandas()
                
                point_ids.extend(points_inside['id'].tolist())
            
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
            self._show_operation_status(f"✅ Labeled {len(point_ids)} points as {self.current_label}")
            if self.verbose:
                print(f"✅ Labeled {len(point_ids)} points as {self.current_label}")
            
            self.update_layers()
            self.update_query_vector()
            
            # Clear the polygon after processing
            self.draw_control.clear()
            self._update_status()
        
        elif action == 'drawstart':
            # Mark that we're starting to draw a polygon
            if self.lasso_mode:
                self.polygon_drawing = True
                self._update_status()
        
        elif action == 'deleted':
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
        
        # Clear classifier state
        self.classifier = None
        self.scaler = None
        self.predictions_gdf = None
        self.classifier_trained = False
        self.show_heatmap = False
        self.normalize_colors = True
        self.highlight_training_points = True
        self.heatmap_toggle.value = False
        self.normalize_heatmap.value = True
        self.highlight_training.value = True
        
        # Clear all map layers
        empty_geojson = {"type": "FeatureCollection", "features": []}
        self.pos_layer.data = empty_geojson
        self.neg_layer.data = empty_geojson
        self.erase_layer.data = empty_geojson
        self.points.data = empty_geojson
        self.heatmap_layer.locations = []
        
        # Hide histogram component
        self.histogram_output.layout.display = 'none'
        
        # Clear histogram output
        with self.histogram_output:
            self.histogram_output.clear_output()
        
        # Hide and clear classifier output
        self.classifier_output.layout.display = 'none'
        with self.classifier_output:
            self.classifier_output.clear_output()
        
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
            self._show_operation_status(f"🔄 Fetching embeddings for {len(missing_ids)} points...")
            if self.verbose:
                print(f"🔄 Fetching embeddings for {len(missing_ids)} points...")
        
        # Process in chunks to avoid memory issues
        for i in range(0, len(missing_ids), chunk_size):
            chunk = missing_ids[i:i + chunk_size]
            
            # Show chunk progress for very large batches
            if len(missing_ids) > chunk_size:
                chunk_num = i // chunk_size + 1
                total_chunks = (len(missing_ids) - 1) // chunk_size + 1
                self._show_operation_status(f"🔄 Processing chunk {chunk_num}/{total_chunks}")
                if self.verbose:
                    print(f"   Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} points)...")
            
            # Build parameterized query for this chunk
            prepared_chunk = self._prepare_ids_for_query(chunk)
            placeholders = ','.join(['?' for _ in prepared_chunk])
            query = f"""
            SELECT id, embedding
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            
            # Fetch as Arrow then convert to pandas
            arrow_table = self.duckdb_connection.execute(query, prepared_chunk).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()
            
            # Cache the embeddings from this chunk
            for _, row in chunk_df.iterrows():
                embedding = np.array(row['embedding'])
                # Ensure consistent string type for point IDs
                point_id = str(row['id'])
                self.cached_embeddings[point_id] = embedding
        
        if len(missing_ids) > 100:
            self._show_operation_status(f"✅ Cached embeddings for {len(missing_ids)} points")
            if self.verbose:
                print(f"✅ Cached embeddings for {len(missing_ids)} points")

    def search_click(self, b):
        """Perform similarity search based on current query vector."""
        if self.query_vector is None:
            if self.verbose:
                print("⚠️ No query vector available. Please add some positive labels first.")
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
        # sql = DatabaseConstants.get_similarity_search_light_query(self.embedding_dim)
        sql = DatabaseConstants.get_similarity_search_query(self.embedding_dim)
        query_params = [query_vec, total_requested]
        
        # Show search progress in status bar
        if all_labeled_ids:
            self._show_operation_status(f"🔍 Searching for {n_neighbors} points (will filter {len(all_labeled_ids)} labeled)...")
            if self.verbose:
                print(f"🔍 Searching for {n_neighbors} similar points (requesting {total_requested}, will filter {len(all_labeled_ids)} labeled points)...")
        else:
            self._show_operation_status(f"🔍 Searching for {n_neighbors} similar points...")
            if self.verbose:
                print(f"🔍 Searching for {n_neighbors} similar points...")
        
        # Fetch as Arrow table then convert only needed columns to pandas
        arrow_table = self.duckdb_connection.execute(sql, query_params).fetch_arrow_table()
        search_results = arrow_table.select(['id', 'geometry_json', 'geometry_wkt', 'distance']).to_pandas()
        
        # Post-filter to exclude labeled points in memory (much safer than DuckDB NOT IN)
        if all_labeled_ids:
            # Convert to string IDs for consistent comparison
            labeled_id_strings = set(str(lid) for lid in all_labeled_ids)
            # Filter out labeled points
            mask = ~search_results['id'].astype(str).isin(labeled_id_strings)
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
            self._show_operation_status(f"✅ Found {filtered_count} similar points (filtered out {filtered_out} labeled)")
            if self.verbose:
                print(f"✅ Found {filtered_count} similar points after filtering out {filtered_out} labeled points")
        else:
            self._show_operation_status(f"✅ Found {filtered_count} similar points")
        
        # Create geometries from WKT
        geometries = [shapely.wkt.loads(row['geometry_wkt']) if row['geometry_wkt'] else None
                     for _, row in search_results_filtered.iterrows()]
        
        # Create detections DataFrame without embeddings (fetched on-demand during labeling)
        self.detections_with_embeddings = gpd.GeoDataFrame({
            'id': search_results_filtered['id'].astype(str).values,  # Ensure string type
            'distance': search_results_filtered['distance'].values,
            'geometry': geometries
        })
        
        # Create GeoJSON for map display with distance-based coloring
        detections_geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        if not search_results_filtered.empty:
            # Calculate distance range for color mapping
            min_distance = search_results_filtered['distance'].min()
            max_distance = search_results_filtered['distance'].max()
            
            for _, row in search_results_filtered.iterrows():
                # Calculate color based on distance
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
        
        # Update the map with distance-colored points
        self._update_search_layer_with_colors(detections_geojson)
        
        # Generate and display histogram
        if not search_results_filtered.empty:
            distances = search_results_filtered['distance'].values
            self._generate_distance_histogram(distances)
            self.histogram_output.layout.display = 'flex'
        else:
            self.histogram_output.layout.display = 'none'

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        # Don't process clicks when in polygon mode or actively drawing
        if not self.execute_label_point or self.lasso_mode or self.polygon_drawing:
            return
        
        action = kwargs.get('type') 
        if action not in ['click']:
            return
                 
        lat, lon = kwargs.get('coordinates')
        
        clicked_point = Point(lon, lat)
        point_id = None
        
        # First check if we have cached detections
        if self.detections_with_embeddings is not None and len(self.detections_with_embeddings) > 0:
            # Find nearest point in cached detections
            distances = self.detections_with_embeddings.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()
            
            # Use a threshold to ensure we're clicking on an actual point
            if distances[nearest_idx] < UIConstants.CLICK_THRESHOLD:
                nearest_detection = self.detections_with_embeddings.loc[nearest_idx]
                point_id = str(nearest_detection['id'])  # Ensure string type
        
        # If not found in cache, query the database
        if point_id is None:
            # Use lightweight query without embedding
            sql = DatabaseConstants.NEAREST_POINT_LIGHT_QUERY
            
            arrow_table = self.duckdb_connection.execute(sql, [lon, lat]).fetch_arrow_table()
            nearest_result = arrow_table.to_pandas()
            
            if nearest_result.empty:
                return
            
            point_id = str(nearest_result.iloc[0]['id'])  # Convert to string
        
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
            erase_result = self.duckdb_connection.execute(erase_query, [prepared_point_id]).fetchone()
            if erase_result:
                erase_geojson = {
                    "type": "FeatureCollection", 
                    "features": [{
                        "type": "Feature", 
                        "geometry": json.loads(erase_result[0]),
                        "properties": {}
                    }]
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
            props = feature.get('properties', {})
            return {
                'color': 'black',
                'radius': UIConstants.SEARCH_POINT_RADIUS,
                'fillColor': props.get('fillColor', UIConstants.SEARCH_COLOR),
                'opacity': UIConstants.POINT_OPACITY,
                'fillOpacity': UIConstants.POINT_FILL_OPACITY,
                'weight': UIConstants.SEARCH_POINT_WEIGHT
            }
        
        # Update the points layer with new data and style function
        self.points.data = geojson_data
        self.points.style_callback = style_function

    def update_layers(self):
        if self.pos_ids:
            # Prepare IDs for database query
            prepared_pos_ids = self._prepare_ids_for_query(self.pos_ids)
            placeholders = ','.join(['?' for _ in prepared_pos_ids])
            pos_query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            pos_results = self.duckdb_connection.execute(pos_query, prepared_pos_ids).df()
            pos_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in pos_results.iterrows():
                pos_geojson["features"].append({
                    "type": "Feature", 
                    "geometry": json.loads(row['geometry']),
                    "properties": {}
                })
            self.pos_layer.data = pos_geojson
        else:
            self.pos_layer.data = {"type": "FeatureCollection", "features": []}
        
        if self.neg_ids:
            # Prepare IDs for database query
            prepared_neg_ids = self._prepare_ids_for_query(self.neg_ids)
            placeholders = ','.join(['?' for _ in prepared_neg_ids])
            neg_query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            neg_results = self.duckdb_connection.execute(neg_query, prepared_neg_ids).df()
            neg_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in neg_results.iterrows():
                neg_geojson["features"].append({
                    "type": "Feature", 
                    "geometry": json.loads(row['geometry']),
                    "properties": {}
                })
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
        pos_embeddings = [self.cached_embeddings[pid] for pid in self.pos_ids 
                         if pid in self.cached_embeddings]
        
        if not pos_embeddings:
            self.query_vector = None
            return
        
        pos_vec = np.mean(pos_embeddings, axis=0)
        
        # Get negative embeddings from cache
        neg_embeddings = [self.cached_embeddings[nid] for nid in self.neg_ids 
                         if nid in self.cached_embeddings]
        
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
        placeholders = ','.join(['?' for _ in prepared_labeled_ids])
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
            point_id = str(row['id'])  # Ensure string type for consistency
            
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
                embedding = np.array(row['embedding'])
            
            # Create feature with properties including label and embedding
            feature = {
                "type": "Feature",
                "geometry": json.loads(row['geometry_json']),
                "properties": {
                    "id": point_id,
                    "label": label,
                    "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
                }
            }
            features.append(feature)
        
        # Create GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "positive_points": len([f for f in features if f['properties']['label'] == UIConstants.POSITIVE_LABEL]),
                "negative_points": len([f for f in features if f['properties']['label'] == UIConstants.NEGATIVE_LABEL]),
                "embedding_dimension": self.embedding_dim
            }
        }
        
        # Save to file
        filename = f"labeled_dataset_{timestamp}.geojson"
        
        try:
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            # Create summary
            pos_count = len([f for f in features if f['properties']['label'] == UIConstants.POSITIVE_LABEL])
            neg_count = len([f for f in features if f['properties']['label'] == UIConstants.NEGATIVE_LABEL])
            
            if self.verbose:
                print(f"✅ Dataset saved successfully!")
                print(f"📄 Filename: {filename}")
                print(f"📊 Summary:")
                print(f"   - Total points: {len(features)}")
                print(f"   - Positive labels: {pos_count}")
                print(f"   - Negative labels: {neg_count}")
                print(f"   - Embedding dimension: {self.embedding_dim}")
            
            # Optional: Also save a separate CSV with just IDs and labels for easier processing
            labels_df = pd.DataFrame([
                {'id': f['properties']['id'], 'label': f['properties']['label']} 
                for f in features
            ])
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
            with open(filename, 'r') as f:
                geojson_data = json.load(f)
            
            # Clear current labels
            self.pos_ids = []
            self.neg_ids = []
            self.cached_embeddings = {}
            
            # Process features
            for feature in geojson_data['features']:
                point_id = str(feature['properties']['id'])  # Ensure string type
                label = feature['properties']['label']
                embedding = np.array(feature['properties']['embedding'])
                
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
            metadata = geojson_data.get('metadata', {})
            if self.verbose:
                print(f"✅ Dataset loaded successfully!")
                print(f"📊 Summary:")
                print(f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}")
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
            if filename.lower().endswith('.geojson'):
                # Parse GeoJSON
                geojson_data = json.loads(content_bytes.decode('utf-8'))
                self._process_geojson_data(geojson_data, filename)
                
            elif filename.lower().endswith('.parquet'):
                # Parse GeoParquet using pandas/geopandas
                import io
                gdf = gpd.read_parquet(io.BytesIO(content_bytes))
                self._process_geoparquet_data(gdf, filename)
                
            else:
                raise ValueError(f"Unsupported file format. Please use .geojson or .parquet files.")
                
        except Exception as e:
            raise Exception(f"Error processing {filename}: {str(e)}")

    def _process_geojson_data(self, geojson_data, filename):
        """Process GeoJSON data and populate labels."""
        # Clear current labels
        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}
        
        # Process features
        for feature in geojson_data['features']:
            point_id = str(feature['properties']['id'])  # Ensure string type
            label = feature['properties']['label']
            embedding = np.array(feature['properties']['embedding'])
            
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
        metadata = geojson_data.get('metadata', {})
        if self.verbose:
            print(f"✅ Dataset loaded successfully from {filename}!")
            print(f"📊 Summary:")
            print(f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}")
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
        required_cols = ['id', 'label', 'embedding']
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Required column '{col}' not found in {filename}")
        
        # Process each row
        for _, row in gdf.iterrows():
            point_id = str(row['id'])  # Ensure string type
            label = row['label']
            
            # Handle embedding - could be stored as array or list
            if isinstance(row['embedding'], (list, np.ndarray)):
                embedding = np.array(row['embedding'])
            else:
                # Try to parse if it's stored as string
                embedding = np.array(json.loads(row['embedding']))
            
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
            print(f"📊 Summary:")
            print(f"   - Total points: {len(gdf)}")
            print(f"   - Positive labels: {len(self.pos_ids)}")
            print(f"   - Negative labels: {len(self.neg_ids)}")

    def _update_basemap_button_styles(self):
        """Update basemap button styles to highlight current selection."""
        for basemap_name, btn in self.basemap_buttons.items():
            if basemap_name == self.current_basemap:
                btn.button_style = 'info'  # Blue highlight for active
            else:
                btn.button_style = ''  # Default style

    def _generate_distance_histogram(self, distances: np.ndarray) -> None:
        """Generate and display distance histogram."""
        with self.histogram_output:
            self.histogram_output.clear_output(wait=True)
            
            if len(distances) == 0:
                return
            
            plt.figure(figsize=(8, 3))
            n_bins = min(30, len(distances) // 10 + 5)  # Adaptive bin count
            plt.hist(distances, bins=n_bins, alpha=0.7, color='lightblue', edgecolor='black')
            
            plt.xlabel('Distance', fontsize=12)
            plt.tick_params(axis='x', labelsize=11)  # Bigger x-axis ticks
            plt.gca().set_yticklabels([])  # Remove y-axis labels
            plt.title(f'Distance Distribution ({len(distances)} points)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def classify_all(self, b):
        """Train linear classifier on labeled data and apply to all embeddings."""
        # Check if we have sufficient labeled data
        if len(self.pos_ids) == 0:
            if self.verbose:
                print("⚠️ No positive labels found. Please add some positive labels first.")
            return
        
        if len(self.pos_ids) + len(self.neg_ids) < 3:
            if self.verbose:
                print("⚠️ Need at least 3 labeled points (positive + negative) to train classifier.")
            return
        
        self._show_operation_status("🤖 Training linear classifier...")
        if self.verbose:
            print("🤖 Training linear classifier on labeled data...")
        
        try:
            # Fetch embeddings for all labeled points
            all_labeled_ids = self.pos_ids + self.neg_ids
            self._fetch_embeddings(all_labeled_ids)
            
            # Prepare training data
            X_train = []
            y_train = []
            
            # Add positive examples
            for pid in self.pos_ids:
                if pid in self.cached_embeddings:
                    X_train.append(self.cached_embeddings[pid])
                    y_train.append(1)
            
            # Add negative examples
            for nid in self.neg_ids:
                if nid in self.cached_embeddings:
                    X_train.append(self.cached_embeddings[nid])
                    y_train.append(0)
            
            if len(X_train) < 2:
                if self.verbose:
                    print("⚠️ Insufficient training data after fetching embeddings.")
                return
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train classifier
            # self.scaler = StandardScaler()
            # X_train_scaled = self.scaler.fit_transform(X_train)
            
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.classifier.fit(X_train, y_train)
            
            # Calculate training accuracy
            train_score = self.classifier.score(X_train, y_train)
            
            self._show_operation_status(f"✅ Classifier trained (accuracy: {train_score:.2f})")
            if self.verbose:
                print(f"✅ Classifier trained successfully!")
                print(f"   Training accuracy: {train_score:.2f}")
                print(f"   Training data: {len(self.pos_ids)} positive, {len(self.neg_ids)} negative")
            
            # Show classifier info
            with self.classifier_output:
                self.classifier_output.clear_output(wait=True)
                print(f"🤖 Linear Classifier Summary:")
                print(f"   Training accuracy: {train_score:.2f}")
                print(f"   Positive examples: {len(self.pos_ids)}")
                print(f"   Negative examples: {len(self.neg_ids)}")
                print(f"   Feature dimension: {X_train.shape[1]}")
            
            self.classifier_output.layout.display = 'flex'
            self.classifier_trained = True
            
            # Apply classifier to all points
            self._apply_classifier_to_all()
            
        except Exception as e:
            self._show_operation_status(f"❌ Classifier training failed")
            if self.verbose:
                print(f"❌ Error training classifier: {str(e)}")

    def _apply_classifier_to_all(self, chunk_size=5000):
        """Apply trained classifier to all embeddings in the database."""
        if not self.classifier_trained or self.classifier is None:
            return
        
        self._show_operation_status("🔄 Applying classifier to all points...")
        if self.verbose:
            print("🔄 Applying classifier to all points in database...")
        
        try:
            # Get total count for progress tracking
            count_query = "SELECT COUNT(*) FROM geo_embeddings"
            total_points = self.duckdb_connection.execute(count_query).fetchone()[0]
            
            if self.verbose:
                print(f"   Processing {total_points} points...")
            
            # Process in chunks to avoid memory issues
            predictions_list = []
            confidence_list = []
            geometries_list = []
            ids_list = []
            
            offset = 0
            processed = 0
            
            while offset < total_points:
                # Show progress
                progress_pct = (offset / total_points) * 100
                self._show_operation_status(f"🔄 Classifying... {progress_pct:.0f}% ({offset}/{total_points})")
                
                # Fetch chunk of data
                chunk_query = f"""
                SELECT id, embedding, ST_AsGeoJSON(geometry) as geometry_json, ST_AsText(geometry) as geometry_wkt
                FROM geo_embeddings 
                LIMIT {chunk_size} OFFSET {offset}
                """
                
                arrow_table = self.duckdb_connection.execute(chunk_query).fetch_arrow_table()
                chunk_df = arrow_table.to_pandas()
                
                if chunk_df.empty:
                    break
                
                # Extract embeddings
                embeddings = np.array([np.array(emb) for emb in chunk_df['embedding']])
                
                # Scale embeddings
                # embeddings_scaled = self.scaler.transform(embeddings)
                
                # Get predictions and confidence scores
                predictions = self.classifier.predict(embeddings)
                confidence_scores = self.classifier.predict_proba(embeddings)[:, 1]  # Probability of positive class
                
                # Store results
                predictions_list.extend(predictions)
                confidence_list.extend(confidence_scores)
                ids_list.extend(chunk_df['id'].astype(str).tolist())
                
                # Create geometries from WKT
                geometries = [shapely.wkt.loads(wkt) if wkt else None for wkt in chunk_df['geometry_wkt']]
                geometries_list.extend(geometries)
                
                offset += len(chunk_df)
                processed += len(chunk_df)
                
                # Update progress more frequently for large datasets
                if processed % 10000 == 0 or processed >= total_points:
                    progress_pct = (processed / total_points) * 100
                    self._show_operation_status(f"🔄 Classified {processed}/{total_points} points ({progress_pct:.0f}%)")
            
            # Create GeoDataFrame with predictions
            self.predictions_gdf = gpd.GeoDataFrame({
                'id': ids_list,
                'prediction': predictions_list,
                'confidence': confidence_list,
                'geometry': geometries_list
            })
            
            # Update heatmap if enabled
            if self.show_heatmap:
                self._update_heatmap_layer()
            
            # Show completion status
            n_positive = np.sum(predictions_list)
            n_negative = len(predictions_list) - n_positive
            avg_confidence = np.mean(confidence_list)
            
            self._show_operation_status(f"✅ Classified {processed} points ({n_positive} positive, {n_negative} negative)")
            
            if self.verbose:
                print(f"✅ Classification complete!")
                print(f"   Total points: {processed}")
                print(f"   Predicted positive: {n_positive}")
                print(f"   Predicted negative: {n_negative}")
                print(f"   Average confidence: {avg_confidence:.3f}")
            
            # Update classifier output with results
            with self.classifier_output:
                self.classifier_output.clear_output(wait=True)
                print(f"🤖 Classification Results:")
                print(f"   Total points classified: {processed}")
                print(f"   Predicted positive: {n_positive}")
                print(f"   Predicted negative: {n_negative}")
                print(f"   Average confidence: {avg_confidence:.3f}")
                
                # Show confidence distribution statistics
                conf_min = np.min(confidence_list)
                conf_max = np.max(confidence_list)
                conf_25 = np.percentile(confidence_list, 25)
                conf_75 = np.percentile(confidence_list, 75)
                print(f"   Confidence range: {conf_min:.3f} - {conf_max:.3f}")
                print(f"   Confidence 25%-75%: {conf_25:.3f} - {conf_75:.3f}")
                
                # Check confidence for training points
                if self.pos_ids or self.neg_ids:
                    train_confidences = []
                    for train_id in (self.pos_ids + self.neg_ids):
                        idx = self.predictions_gdf[self.predictions_gdf['id'] == train_id].index
                        if len(idx) > 0:
                            train_conf = self.predictions_gdf.loc[idx[0], 'confidence']
                            train_confidences.append(train_conf)
                    
                    if train_confidences:
                        avg_train_conf = np.mean(train_confidences)
                        print(f"   Training points avg confidence: {avg_train_conf:.3f}")
                
                # Generate confidence histogram
                plt.figure(figsize=(8, 3))
                plt.hist(confidence_list, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                plt.xlabel('Confidence Score', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.title(f'Confidence Distribution ({len(confidence_list)} points)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                print(f"   Toggle 'Show Heatmap' to visualize smooth confidence map")
            
        except Exception as e:
            self._show_operation_status(f"❌ Classification failed")
            if self.verbose:
                print(f"❌ Error applying classifier: {str(e)}")

    def _update_heatmap_layer(self):
        """Update the heatmap layer with confidence-based smooth heatmap."""
        if self.predictions_gdf is None or len(self.predictions_gdf) == 0:
            return
        
        if self.verbose:
            print("🎨 Updating smooth heatmap layer...")
        
        # Sample points for performance if dataset is very large
        sample_size = 15000  # Increased sample size for smoother heatmap
        if len(self.predictions_gdf) > sample_size:
            display_df = self.predictions_gdf.sample(n=sample_size, random_state=42)
            if self.verbose:
                print(f"   Sampling {sample_size} points for heatmap display")
        else:
            display_df = self.predictions_gdf
        
        # Get confidence values for normalization
        confidences = display_df['confidence'].values.copy()  # Make a copy so we can modify it
        
        # Highlight training points with maximum confidence if enabled
        if self.highlight_training_points and (self.pos_ids or self.neg_ids):
            training_ids = set(str(tid) for tid in (self.pos_ids + self.neg_ids))
            for idx, (df_idx, row) in enumerate(display_df.iterrows()):
                if str(row['id']) in training_ids:
                    # Positive labels get confidence 1.0, negative labels get confidence 0.0
                    if str(row['id']) in [str(pid) for pid in self.pos_ids]:
                        confidences[idx] = 1.0
                    else:  # Negative labels
                        confidences[idx] = 0.0
            
            if self.verbose:
                n_highlighted = len([row for _, row in display_df.iterrows() if str(row['id']) in training_ids])
                print(f"   Highlighted {n_highlighted} training points")
        
        # Use percentile-based coloring or absolute confidence based on checkbox setting
        if self.normalize_colors:
            # Use percentile-based coloring (new default behavior)
            if len(confidences) > 1:
                # Calculate percentile ranks for each confidence value
                # This maps each value to its percentile rank (0-1)
                percentile_ranks = rankdata(confidences, method='average') / len(confidences)
                
                if self.verbose:
                    conf_min, conf_max = np.min(confidences), np.max(confidences)
                    p20 = np.percentile(confidences, 20)
                    p50 = np.percentile(confidences, 50) 
                    p80 = np.percentile(confidences, 80)
                    print(f"   Original confidence range: [{conf_min:.3f}, {conf_max:.3f}]")
                    print(f"   Confidence percentiles: 20th={p20:.3f}, 50th={p50:.3f}, 80th={p80:.3f}")
                    print(f"   Using percentile-based coloring (upper 20% = darkest)")
                
                normalized_confidences = percentile_ranks
            else:
                # Single value case
                normalized_confidences = np.full_like(confidences, 0.5)
                if self.verbose:
                    print(f"   Single confidence value: {confidences[0]:.3f}, using 0.5")
        else:
            # Use absolute confidence values (original behavior)
            if len(confidences) > 1:
                conf_min = np.min(confidences)
                conf_max = np.max(confidences)
                
                if conf_max > conf_min:
                    # Stretch to 0-1 range for consistent heatmap intensity
                    normalized_confidences = (confidences - conf_min) / (conf_max - conf_min)
                    if self.verbose:
                        print(f"   Using absolute confidence, normalized from [{conf_min:.3f}, {conf_max:.3f}] to [0, 1]")
                else:
                    # All values are the same
                    normalized_confidences = np.full_like(confidences, 0.5)
                    if self.verbose:
                        print(f"   All confidence values are {conf_min:.3f}, using 0.5")
            else:
                # Single value case
                normalized_confidences = np.full_like(confidences, 0.5)
                if self.verbose:
                    print(f"   Single confidence value: {confidences[0]:.3f}, using 0.5")
        
        # Create location points for HeatmapLayer: [lat, lon, intensity]
        heatmap_points = []
        
        for i, (_, row) in enumerate(display_df.iterrows()):
            # Get lat/lon from geometry
            if row['geometry'] is not None:
                if hasattr(row['geometry'], 'coords'):
                    # Point geometry
                    lon, lat = row['geometry'].coords[0]
                elif hasattr(row['geometry'], 'x'):
                    # Point geometry (alternative access)
                    lon, lat = row['geometry'].x, row['geometry'].y
                else:
                    continue
                
                # Use percentile rank as intensity
                intensity = float(normalized_confidences[i])
                
                # Add point as [lat, lon, intensity]
                heatmap_points.append([lat, lon, intensity])
        
        # Update heatmap layer
        self.heatmap_layer.locations = heatmap_points
        
        if self.verbose:
            final_min, final_max = (np.min(normalized_confidences), np.max(normalized_confidences))
            mode = "percentile-based" if self.normalize_colors else "absolute confidence"
            print(f"✅ Smooth heatmap updated with {len(heatmap_points)} points using {mode} coloring")
            print(f"   Final intensity range: [{final_min:.3f}, {final_max:.3f}]")

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_owns_connection') and self._owns_connection:
            if hasattr(self, 'duckdb_connection') and self.duckdb_connection:
                self.duckdb_connection.close()
                if self.verbose:
                    print("🔌 DuckDB connection closed.")

    def _update_legend(self):
        """Update legend dynamically based on current coloring mode."""
        if self.normalize_colors:
            heatmap_description = """<div style='margin-top: 3px;'><strong>Confidence Heatmap (Percentile-Based):</strong> 
                    <span style='color: #0000ff; font-weight: bold;'>🔵 Bottom 20%</span> → 
                    <span style='color: #00ffff; font-weight: bold;'>🔷 Lower-Middle</span> → 
                    <span style='color: #ffff00; font-weight: bold;'>🟡 Upper-Middle</span> → 
                    <span style='color: #ff0000; font-weight: bold;'>🔴 Top 20%</span>
                    <br/><span style='font-size: 10px; color: #666;'>Colors based on percentile ranks, not absolute confidence</span>
                </div>"""
        else:
            heatmap_description = """<div style='margin-top: 3px;'><strong>Confidence Heatmap (Absolute Values):</strong> 
                    <span style='color: #0000ff; font-weight: bold;'>🔵 Low</span> → 
                    <span style='color: #00ffff; font-weight: bold;'>🔷 Medium-Low</span> → 
                    <span style='color: #ffff00; font-weight: bold;'>🟡 Medium-High</span> → 
                    <span style='color: #ff0000; font-weight: bold;'>🔴 High</span>
                    <br/><span style='font-size: 10px; color: #666;'>Colors based on actual confidence values, normalized to 0-1</span>
                </div>"""
        
        if not hasattr(self, 'legend'):
            self.legend = HTML()
        
        self.legend.value = f"""
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
                {heatmap_description}
            </div>
        """

    def _on_normalize_heatmap(self, change):
        """Handle heatmap normalization toggle."""
        self.normalize_colors = change['new']
        self._update_legend()  # Update legend to reflect new mode
        if self.show_heatmap and self.predictions_gdf is not None:
            self._update_heatmap_layer()
