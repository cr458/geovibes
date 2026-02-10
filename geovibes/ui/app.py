"""GeoVibes ipyleaflet application orchestrator."""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.ops
import shapely.wkt
import webbrowser
import faiss
import pyproj
from IPython.display import display
from ipywidgets import (
    Button,
    FileUpload,
    HTML,
    Layout,
    VBox,
)
import ipyvuetify as v

from geovibes.ui_config import BasemapConfig, UIConstants
from geovibes.ui.data_manager import DataManager
from geovibes.ui.datasets import DatasetManager
from geovibes.ui.map_manager import MapManager
from geovibes.ui.state import AppState
from geovibes.ui.status import StatusBus
from geovibes.ui.tiles import TilePanel
from geovibes.ui.utils import log_to_file

warnings.simplefilter("ignore", category=FutureWarning)

SIDE_PANEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.geovibes-panel,
.geovibes-panel * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Make search button more prominent */
.geovibes-panel .search-btn {
    height: 40px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.geovibes-panel .v-btn {
    text-transform: none !important;
    letter-spacing: 0.3px !important;
    font-size: 12px !important;
}

.geovibes-panel .v-btn__content {
    font-weight: 500 !important;
}

.geovibes-panel .section-label {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: block;
}

.geovibes-panel .v-card {
    margin-bottom: 8px !important;
}

.geovibes-panel .v-btn-toggle {
    width: 100%;
}

.geovibes-panel .v-btn-toggle .v-btn {
    flex: 1 !important;
    height: 32px !important;
}

.geovibes-panel .v-slider {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.geovibes-panel .v-select {
    font-size: 12px !important;
}

.geovibes-panel .v-select .v-input__slot {
    min-height: 36px !important;
}

.geovibes-panel .v-select .v-select__selection {
    font-size: 12px !important;
}

.geovibes-panel .v-list-item__title {
    font-size: 12px !important;
}

.geovibes-panel .text-body-2 {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* Compact FileUpload widget */
.geovibes-panel .widget-upload {
    padding: 0 !important;
    margin: 4px 0 !important;
}

.geovibes-panel .widget-upload > .widget-label {
    display: none !important;
}

.geovibes-panel .widget-upload-label {
    font-size: 11px !important;
    padding: 4px 8px !important;
    margin: 0 !important;
}
</style>
"""

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn(
        "MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key.",
        RuntimeWarning,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


FAISS_NPROBE_DEFAULT = max(1, _env_int("GEOVIBES_FAISS_NPROBE", 1024))
FAISS_NPROBE_HIGH = max(FAISS_NPROBE_DEFAULT, _env_int("GEOVIBES_FAISS_NPROBE_HIGH", 2048))
FAISS_NPROBE_HIGH_THRESHOLD = max(
    1, _env_int("GEOVIBES_FAISS_NPROBE_HIGH_THRESHOLD", 5000)
)
PREFETCH_FAST_K_DEFAULT = max(1, _env_int("GEOVIBES_PREFETCH_FAST_K", 100))
PREFETCH_TOPK_MIN = max(1, _env_int("GEOVIBES_PREFETCH_TOPK_MIN", 100))
PREFETCH_TOPK_MAX = max(PREFETCH_TOPK_MIN, _env_int("GEOVIBES_PREFETCH_TOPK_MAX", 200))
PREFETCH_TOPK_RATIO = max(0.01, _env_float("GEOVIBES_PREFETCH_TOPK_RATIO", 0.2))
PREFETCH_WORKER_TARGET = max(1, _env_int("GEOVIBES_PREFETCH_WORKER_TARGET", 44))
PREFETCH_WORKER_CAP = max(PREFETCH_WORKER_TARGET, _env_int("GEOVIBES_PREFETCH_WORKER_CAP", 44))
PREFETCH_WORKER_FLOOR = max(1, _env_int("GEOVIBES_PREFETCH_WORKER_FLOOR", 16))
PREFETCH_WORKER_MEM_MB = max(16, _env_int("GEOVIBES_PREFETCH_WORKER_MEM_MB", 96))
EMBEDDING_LRU_SIZE = max(0, _env_int("GEOVIBES_EMBEDDING_LRU_SIZE", 2000))


def group_databases_by_region(databases: List[Dict]) -> Dict[str, List[Dict]]:
    """Group database entries by their region.

    Args:
        databases: List of database entry dictionaries with 'region' key

    Returns:
        Dictionary mapping region names to lists of database entries
    """
    grouped: Dict[str, List[Dict]] = {}
    for db in databases:
        region = db.get("region", "Other")
        if region not in grouped:
            grouped[region] = []
        grouped[region].append(db)
    return grouped


def build_grouped_dropdown_items(
    databases: List[Dict], include_placeholder: bool = False
) -> List[Dict]:
    """Build dropdown items with region headers and dividers.

    Args:
        databases: List of database entry dictionaries
        include_placeholder: If True, add "Select a database..." placeholder at start

    Returns:
        List of dropdown items with headers, items, and dividers
    """
    items: List[Dict] = []

    if include_placeholder:
        items.append({"text": "Select a database...", "value": None, "disabled": True})

    grouped = group_databases_by_region(databases)

    # Sort regions alphabetically
    sorted_regions = sorted(grouped.keys())

    for i, region in enumerate(sorted_regions):
        # Add divider before all regions except the first (and after placeholder if present)
        if i > 0:
            items.append({"divider": True})

        # Add header with capitalized region name
        items.append({"header": region.title()})

        # Add database items for this region
        for db in grouped[region]:
            display_name = db.get("display_name", db["db_path"])
            if db.get("is_remote"):
                display_name = f"Remote / {display_name}"
            else:
                display_name = f"Local / {display_name}"
            items.append({"text": display_name, "value": db["db_path"]})

    return items


class GeoVibes:
    """Interactive map interface for geospatial similarity search."""

    @classmethod
    def create(
        cls,
        duckdb_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        geometry_cache_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2025-01-01",
        verbose: bool = False,
        enable_ee: Optional[bool] = None,
        include_remote: bool = False,
        **kwargs,
    ):
        return cls(
            duckdb_path=duckdb_path,
            faiss_path=faiss_path,
            geometry_cache_path=geometry_cache_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            enable_ee=enable_ee,
            include_remote=include_remote,
            **kwargs,
        )

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        geometry_cache_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        duckdb_connection=None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        baselayer_url: Optional[str] = None,
        disable_ee: bool = False,
        verbose: bool = False,
        enable_ee: Optional[bool] = None,
        include_remote: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes...")

        if "enable_ee" in unused_kwargs and self.verbose:
            print(
                "ℹ️ Pass enable_ee via config or GEOVIBES_ENABLE_EE environment variable."
            )

        # Core services
        self.data = DataManager(
            duckdb_path=duckdb_path,
            faiss_path=faiss_path,
            geometry_cache_path=geometry_cache_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            config=config,
            config_path=config_path,
            duckdb_connection=duckdb_connection,
            baselayer_url=baselayer_url,
            disable_ee=disable_ee,
            verbose=verbose,
            enable_ee=enable_ee,
            include_remote=include_remote,
        )
        self.id_column_candidates = getattr(self.data, "id_column_candidates", ["id"])
        self.external_id_column = getattr(self.data, "external_id_column", "id")
        self.state = AppState()
        self._prefetch_generation = 0
        self._prefetch_generation_lock = threading.Lock()
        self._embedding_lru_size = EMBEDDING_LRU_SIZE
        self.status_bus = StatusBus()
        self.map_manager = MapManager(
            data_manager=self.data,
            state=self.state,
            status_bus=self.status_bus,
            verbose=self.verbose,
        )
        self.dataset_manager = DatasetManager(
            data_manager=self.data,
            map_manager=self.map_manager,
            state=self.state,
            verbose=self.verbose,
        )
        self.tile_panel = TilePanel(
            state=self.state,
            map_manager=self.map_manager,
            on_label=self._handle_tile_label,
            on_center=self._handle_tile_center,
            verbose=self.verbose,
        )

        self._build_ui()
        self._wire_events()

        self.map_manager.update_boundary_layer(self.data.effective_boundary_path)
        self._update_layers()
        self._show_operation_status("Ready")
        self._update_status()

        display(self.main_layout)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.side_panel, self.ui_widgets = self._build_side_panel()
        self.main_layout = self.map_manager.make_layout(self.side_panel)

    def _build_side_panel(self):
        css_widget = HTML(SIDE_PANEL_CSS)

        # Search section with ipyvuetify (style L: full-width search + icon button)
        # Disabled until a database is connected (deferred loading mode)
        search_disabled = not self.data.is_connected()
        self.search_btn = v.Btn(
            block=True,
            color="primary",
            depressed=True,
            disabled=search_disabled,
            class_="search-btn",
            children=[
                v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
                "Search" if not search_disabled else "Select Database",
            ],
        )
        self.tiles_button = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-view-grid-outline"])],
        )
        search_row = v.Row(
            no_gutters=True,
            align="center",
            class_="mb-2",
            children=[
                v.Col(cols=10, class_="pr-1", children=[self.search_btn]),
                v.Col(
                    cols=2,
                    class_="pl-1 d-flex justify-end",
                    children=[self.tiles_button],
                ),
            ],
        )

        self.neighbors_slider = v.Slider(
            v_model=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            thumb_label=True,  # Only show on drag, not always
            hide_details=True,
            class_="mt-0 flex-grow-1",
        )
        self.neighbors_label = v.Html(
            tag="span",
            class_="text-body-2 font-weight-medium ml-2",
            children=[str(UIConstants.DEFAULT_NEIGHBORS)],
            style_="min-width: 45px; text-align: right;",
        )
        slider_row = v.Row(
            no_gutters=True,
            align="center",
            children=[
                v.Col(cols=10, class_="pa-0", children=[self.neighbors_slider]),
                v.Col(
                    cols=2,
                    class_="pa-0 d-flex justify-end",
                    children=[self.neighbors_label],
                ),
            ],
        )

        search_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[search_row, slider_row],
        )

        # Label toggle using ipyvuetify BtnToggle with MDI icons
        self._label_values = ["Positive", "Negative", "Erase"]
        self.label_toggle = v.BtnToggle(
            v_model=0,
            mandatory=True,
            class_="d-flex",
            children=[
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-thumb-up-outline"])],
                ),
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-thumb-down-outline"])],
                ),
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-eraser"])],
                ),
            ],
        )

        label_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["LABEL"]),
                self.label_toggle,
            ],
        )

        # Mode toggle with ipyvuetify BtnToggle
        self._mode_values = ["point", "polygon"]
        self.selection_mode = v.BtnToggle(
            v_model=0,
            mandatory=True,
            class_="d-flex",
            children=[
                v.Btn(small=True, children=["• Point"]),
                v.Btn(small=True, children=["▢ Polygon"]),
            ],
        )

        mode_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["MODE"]),
                self.selection_mode,
            ],
        )

        # Detection controls using ipyvuetify (same pattern as neighbors_slider)
        self.detection_threshold_slider = v.Slider(
            v_model=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            thumb_label=True,
            hide_details=True,
            class_="mt-0 flex-grow-1",
        )
        self.detection_threshold_label = v.Html(
            tag="span",
            class_="text-body-2 font-weight-medium ml-2",
            children=["0.50"],
            style_="min-width: 45px; text-align: right;",
        )
        detection_slider_row = v.Row(
            no_gutters=True,
            align="center",
            children=[
                v.Col(
                    cols=10, class_="pa-0", children=[self.detection_threshold_slider]
                ),
                v.Col(
                    cols=2,
                    class_="pa-0 d-flex justify-end",
                    children=[self.detection_threshold_label],
                ),
            ],
        )
        self.detection_controls = v.Card(
            outlined=True,
            class_="section-card pa-3",
            style_="display: none;",
            children=[
                v.Html(
                    tag="span",
                    class_="section-label",
                    children=["DETECTION THRESHOLD"],
                ),
                detection_slider_row,
            ],
        )

        # Database dropdown with ipyvuetify (grouped by region)
        if getattr(self.data, "available_databases", []):
            include_placeholder = not self.data.is_connected()
            db_items = build_grouped_dropdown_items(
                self.data.available_databases, include_placeholder=include_placeholder
            )
            # Start with no selection in deferred mode, or current path otherwise
            initial_value = (
                None
                if not self.data.is_connected()
                else self.data.current_database_path
            )
            self.database_dropdown = v.Select(
                v_model=initial_value,
                items=db_items,
                dense=True,
                outlined=True,
                hide_details=True,
                label="Database" if not self.data.is_connected() else None,
            )
        else:
            self.database_dropdown = None

        # Basemap dropdown with ipyvuetify (same style as database dropdown)
        basemap_names = list(self.map_manager.basemap_tiles.keys())
        basemap_items = [
            {"text": name.replace("_", " "), "value": name} for name in basemap_names
        ]
        self.basemap_dropdown = v.Select(
            v_model=basemap_names[0] if basemap_names else None,
            items=basemap_items,
            dense=True,
            outlined=True,
            hide_details=True,
        )
        self.basemap_names = basemap_names

        # Export buttons with ipyvuetify (using MDI outline icons)
        self.save_btn = v.Btn(
            small=True,
            children=[
                v.Icon(small=True, children=["mdi-content-save-outline"]),
                " Save",
            ],
        )
        self.load_btn = v.Btn(
            small=True,
            children=[
                v.Icon(small=True, children=["mdi-folder-open-outline"]),
                " Load",
            ],
        )
        self.file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none", margin="4px 0 0 0"),
        )
        self.add_vector_btn = v.Btn(
            small=True,
            children=[v.Icon(small=True, children=["mdi-vector-polygon"]), " Vector"],
        )
        self.vector_file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none", margin="4px 0 0 0"),
        )
        self.google_maps_btn = v.Btn(
            small=True,
            children=[v.Icon(small=True, children=["mdi-google-maps"]), " Maps"],
        )

        # Database card (always visible)
        if self.database_dropdown:
            database_card = v.Card(
                outlined=True,
                class_="section-card pa-3",
                children=[
                    v.Html(tag="span", class_="section-label", children=["DATABASE"]),
                    self.database_dropdown,
                ],
            )
        else:
            database_card = None

        # Basemaps card (always visible)
        basemaps_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["BASEMAP"]),
                self.basemap_dropdown,
            ],
        )

        # Export & Tools card (always visible)
        # FileUpload widgets are kept outside v.Card to avoid rendering issues
        export_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["EXPORT & TOOLS"]),
                v.BtnToggle(
                    v_model=None,
                    dense=True,
                    class_="d-flex flex-wrap",
                    children=[
                        self.save_btn,
                        self.load_btn,
                    ],
                ),
                v.BtnToggle(
                    v_model=None,
                    dense=True,
                    class_="d-flex flex-wrap mt-1",
                    children=[
                        self.add_vector_btn,
                        self.google_maps_btn,
                    ],
                ),
            ],
        )
        # Container for file uploads (placed outside v.Card for proper rendering)
        self.upload_container = VBox(
            [self.file_upload, self.vector_file_upload],
            layout=Layout(width="100%", padding="0 12px", margin="0"),
        )

        # Keep accordion_container reference for compatibility but not used
        self.accordion_container = VBox(
            [
                w
                for w in [
                    database_card,
                    basemaps_card,
                    export_card,
                    self.upload_container,
                ]
                if w is not None
            ],
            layout=Layout(width="100%"),
        )

        # Reset button
        self.reset_btn = v.Btn(
            block=True,
            color="error",
            outlined=True,
            class_="mt-3 text-none",
            children=[
                v.Icon(small=True, class_="mr-1", children=["mdi-trash-can-outline"]),
                "Reset",
            ],
        )

        # Collapse button (keep ipywidgets for simplicity)
        self.collapse_btn = Button(
            description="◀",
            layout=Layout(
                width=UIConstants.COLLAPSE_BUTTON_SIZE,
                height=UIConstants.COLLAPSE_BUTTON_SIZE,
            ),
            tooltip="Collapse/Expand Panel",
        )
        self.panel_collapsed = False

        # Wrap in VBox with ipyvuetify components
        panel = VBox(
            [
                css_widget,
                search_card,
                label_card,
                mode_card,
                self.detection_controls,
                self.accordion_container,
                self.reset_btn,
            ],
            layout=Layout(
                width=UIConstants.PANEL_WIDTH, padding="8px", overflow="hidden"
            ),
        )
        panel.add_class("geovibes-panel")

        ui_widgets = {
            "search_btn": self.search_btn,
            "reset_btn": self.reset_btn,
            "label_toggle": self.label_toggle,
            "selection_mode": self.selection_mode,
            "neighbors_slider": self.neighbors_slider,
            "basemap_dropdown": self.basemap_dropdown,
            "save_btn": self.save_btn,
            "load_btn": self.load_btn,
            "file_upload": self.file_upload,
            "add_vector_btn": self.add_vector_btn,
            "vector_file_upload": self.vector_file_upload,
            "detection_threshold_slider": self.detection_threshold_slider,
            "google_maps_btn": self.google_maps_btn,
            "collapse_btn": self.collapse_btn,
            "tiles_button": self.tiles_button,
            "database_dropdown": self.database_dropdown,
        }
        return panel, ui_widgets

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        # ipyvuetify buttons use on_event instead of on_click
        self.search_btn.on_event("click", lambda *args: self.search_click(None))
        self.reset_btn.on_event("click", lambda *args: self.reset_all(None))
        self.tiles_button.on_event("click", lambda *args: self.tile_panel.toggle())

        # Label toggle uses v_model (index)
        self.label_toggle.observe(self._on_label_toggle_change, names="v_model")

        # BtnToggle uses v_model (index) instead of value
        self.selection_mode.observe(self._on_selection_mode_change, names="v_model")

        # Slider label update
        self.neighbors_slider.observe(self._on_neighbors_slider_change, names="v_model")

        # Basemap dropdown uses v_model (value)
        self.basemap_dropdown.observe(self._on_basemap_dropdown_change, names="v_model")

        # Database dropdown uses v_model
        if self.database_dropdown:
            self.database_dropdown.observe(self._on_database_change, names="v_model")

        self.collapse_btn.on_click(self._on_toggle_collapse)

        # Export buttons
        self.save_btn.on_event("click", lambda *args: self._handle_save_dataset())
        self.load_btn.on_event(
            "click",
            lambda *args: self._toggle_vuetify_upload(
                self.load_btn,
                self.file_upload,
                [v.Icon(small=True, children=["mdi-close"]), " Cancel"],
                [v.Icon(small=True, children=["mdi-folder-open-outline"]), " Load"],
            ),
        )
        self.file_upload.observe(self._on_file_upload, names="value")
        self.add_vector_btn.on_event(
            "click",
            lambda *args: self._toggle_vuetify_upload(
                self.add_vector_btn,
                self.vector_file_upload,
                [v.Icon(small=True, children=["mdi-close"]), " Cancel"],
                [v.Icon(small=True, children=["mdi-vector-polygon"]), " Vector"],
            ),
        )
        self.vector_file_upload.observe(self._on_vector_upload, names="value")
        self.detection_threshold_slider.observe(
            self._on_detection_threshold_change, names="v_model"
        )
        self.google_maps_btn.on_event(
            "click", lambda *args: self._on_google_maps_click(None)
        )

        self.map_manager.register_draw_handler(self._handle_draw)
        self.map_manager.map.on_interaction(self._on_map_interaction)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_label_toggle_change(self, change) -> None:
        idx = change["new"]
        if idx is not None and 0 <= idx < len(self._label_values):
            value = self._label_values[idx]
            self.state.set_label_mode(value)
            self._update_status()

    def _on_selection_mode_change(self, change) -> None:
        # v_model gives us an index, convert to value
        idx = change["new"]
        if idx is not None and 0 <= idx < len(self._mode_values):
            value = self._mode_values[idx]
            self.state.selection_mode = value
            self.state.lasso_mode = value == "polygon"
            self.state.execute_label_point = value != "polygon"
            self._update_status()

    def _toggle_vuetify_upload(
        self, btn, file_upload, cancel_children, default_children
    ) -> None:
        if file_upload.layout.display == "none":
            file_upload.layout.display = "block"
            btn.children = cancel_children
        else:
            file_upload.layout.display = "none"
            btn.children = default_children

    def _on_neighbors_slider_change(self, change) -> None:
        value = change["new"]
        if value is not None:
            self.neighbors_label.children = [f"{value:,}"]

    def _on_basemap_dropdown_change(self, change) -> None:
        basemap_name = change["new"]
        if basemap_name:
            self.map_manager.update_basemap(basemap_name)
            self.tile_panel.handle_map_basemap_change(basemap_name)

    def _on_toggle_collapse(self, _button) -> None:
        if self.panel_collapsed:
            self.accordion_container.layout.display = "flex"
            self.collapse_btn.description = "◀"
            self.panel_collapsed = False
        else:
            self.accordion_container.layout.display = "none"
            self.collapse_btn.description = "▶"
            self.panel_collapsed = True

    def _on_database_change(self, change) -> None:
        new_path = change["new"]
        # Skip if no selection (placeholder selected) or same as current
        if not new_path or new_path == self.data.current_database_path:
            return

        # Check if this is a remote database
        db_info = self.data.database_info_by_path.get(new_path, {})
        if db_info.get("is_remote"):
            # Use progressive loading for remote databases
            self._start_progressive_loading(db_info)
        else:
            # Synchronous loading for local databases
            self._switch_database_sync(new_path)

    def _switch_database_sync(self, new_path: str) -> None:
        """Synchronously switch to a local database."""
        self._show_operation_status("🔄 Loading database...")
        try:
            # Use connect_to_database for initial connection, switch_database for switching
            if not self.data.is_connected():
                self.data.connect_to_database(new_path)
            else:
                self.data.switch_database(new_path)
            self.id_column_candidates = getattr(
                self.data, "id_column_candidates", ["id"]
            )
            self.external_id_column = getattr(self.data, "external_id_column", "id")
            self.map_manager.center_on(self.data.center_y, self.data.center_x)
            self.map_manager.update_boundary_layer(self.data.effective_boundary_path)
            self.reset_all(clear_overlays=True)
            if self.database_dropdown:
                self.database_dropdown.v_model = new_path
            # Enable search button (was disabled in deferred loading mode)
            self.search_btn.disabled = False
            self.search_btn.children = [
                v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
                "Search",
            ]
        except Exception as exc:
            if self.verbose:
                print(f"❌ Failed to switch database: {exc}")
            self._show_operation_status(f"❌ Failed to load database: {exc}")
        else:
            self._show_operation_status("✅ Database loaded")
        finally:
            self._update_status()

    def _start_progressive_loading(self, db_info: dict) -> None:
        """Start progressive loading for a remote database.

        1. Immediately connect to remote DB
        2. Download FAISS index + geometry cache in background
        3. Enable search when ready
        """
        import threading

        from geovibes.database.faiss_cache import FaissCache

        # Update loading state
        self.state.database_loading = True
        self.state.database_ready = False
        self.state.loading_message = "Connecting..."

        # Disable search button
        self.search_btn.disabled = True
        self.search_btn.children = ["Loading..."]

        self._show_operation_status("📡 Connecting to remote database...")

        db_url = db_info["db_path"]
        faiss_url = db_info["faiss_path"]
        geometry_url = db_info.get("geometry_cache_path")

        def background_loader():
            try:
                # Step 1: Connect to remote DuckDB (fast)
                self.state.loading_message = "Connecting to database..."
                self.data.current_database_path = db_url
                self.data.current_database_info = db_info
                self.data.current_faiss_path = faiss_url
                self.data.current_geometry_cache_path = geometry_url
                self.data.tile_spec = db_info.get("tile_spec")

                # Connect to remote DB
                self.data.duckdb_connection = self.data._connect_duckdb(db_url)
                self.data._apply_duckdb_settings(db_url)

                # Step 2: Download FAISS index (slower)
                self._show_operation_status("📥 Downloading FAISS index...")
                self.state.loading_message = "Downloading FAISS index..."
                cache = FaissCache()
                self.data.faiss_index = cache.get_index(faiss_url, show_progress=True)
                self.data.embedding_dim = self.data._detect_embedding_dim()

                # Step 3: Download geometry cache (if available)
                if geometry_url:
                    self._show_operation_status("📥 Downloading geometry cache...")
                    self.state.loading_message = "Downloading geometry cache..."
                    self.data._load_geometry_cache(geometry_url)

                # Step 4: Warm up database
                self._show_operation_status("🔄 Warming up database...")
                self.state.loading_message = "Warming up..."
                self.data._warm_up_remote_database()

                # Success - update UI on main thread
                self._on_loading_complete(db_info)

            except Exception as e:
                self._on_loading_error(str(e))

        # Start background thread
        thread = threading.Thread(target=background_loader, daemon=True)
        thread.start()

    def _on_loading_complete(self, db_info: dict) -> None:
        """Called when background loading finishes successfully."""
        self.state.database_loading = False
        self.state.database_ready = True
        self.state.loading_message = ""

        # Re-enable search button
        self.search_btn.disabled = False
        self.search_btn.children = [
            v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
            "Search",
        ]

        # Update UI state
        self.id_column_candidates = getattr(self.data, "id_column_candidates", ["id"])
        self.external_id_column = getattr(self.data, "external_id_column", "id")

        # Center map on database region
        self.data.effective_boundary_path, (self.data.center_y, self.data.center_x) = (
            self.data._setup_boundary_and_center()
        )
        self.map_manager.center_on(self.data.center_y, self.data.center_x)
        self.map_manager.update_boundary_layer(self.data.effective_boundary_path)

        # Reset labels and search state
        self.reset_all(clear_overlays=True)

        # Update dropdown selection
        if self.database_dropdown:
            self.database_dropdown.v_model = db_info["db_path"]

        self._show_operation_status("✅ Remote database ready")
        self._update_status()

    def _on_loading_error(self, error_message: str) -> None:
        """Called when background loading fails."""
        self.state.database_loading = False
        self.state.database_ready = False
        self.state.loading_message = ""

        # Re-enable search button (but search won't work)
        self.search_btn.disabled = False
        self.search_btn.children = [
            v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
            "Search",
        ]

        if self.verbose:
            print(f"❌ Failed to load remote database: {error_message}")
        self._show_operation_status(f"❌ Load failed: {error_message}")
        self._update_status()

    def _on_detection_threshold_change(self, change) -> None:
        if not self.state.detection_mode or not self.state.detection_data:
            return
        threshold = change["new"]
        self.detection_threshold_label.children = [f"{threshold:.2f}"]
        self._filter_detection_layer(threshold)
        self._update_detection_tiles()

    def _on_google_maps_click(self, _button) -> None:
        lat, lon = self.map_manager.map.center
        url = f"https://www.google.com/maps/@{lat},{lon},15z"
        webbrowser.open(url, new=2)

    def _on_file_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.reset_all()
            self.dataset_manager.load_from_content(content, file_info["name"])
            if self.state.detection_mode:
                # Show detection controls
                self.detection_controls.style_ = ""
                features = self.state.detection_data.get("features", [])
                num_detections = len(features)

                # Set slider min/max based on dataset probability range
                if features:
                    probs = [
                        f.get("properties", {}).get("probability", 0.5)
                        for f in features
                    ]
                    min_prob = min(probs)
                    max_prob = max(probs)
                    self._detection_prob_min = min_prob
                    self._detection_prob_max = max_prob
                    self.detection_threshold_slider.min = min_prob
                    self.detection_threshold_slider.max = max_prob
                    self.detection_threshold_slider.v_model = min_prob
                    self.detection_threshold_label.children = [f"{min_prob:.2f}"]

                self._show_operation_status(
                    f"🔍 Detection mode: {num_detections} detections loaded. "
                    "Click to label as negative/positive."
                )
                # Apply initial filtering and populate tile panel
                self._filter_detection_layer(self.detection_threshold_slider.v_model)
                self._update_detection_tiles()
            else:
                self.detection_controls.style_ = "display: none;"
                self._update_layers()
                self._update_query_vector()
                self._show_operation_status("✅ Dataset loaded")
        except Exception as exc:
            self._show_operation_status(f"❌ Error loading file: {exc}")
            if self.verbose:
                print(f"❌ Error loading file: {exc}")
        finally:
            self.file_upload.value = ()
            self.file_upload.layout.display = "none"
            self.load_btn.children = [
                v.Icon(small=True, children=["mdi-folder-open-outline"]),
                " Load",
            ]

    def _on_vector_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.dataset_manager.add_vector_from_content(content, file_info["name"])
            self._show_operation_status("✅ Vector layer added")
        except Exception as exc:
            self._show_operation_status(f"❌ Error loading vector: {exc}")
            if self.verbose:
                print(f"❌ Error loading vector: {exc}")
        finally:
            self.vector_file_upload.value = ()
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.children = [
                v.Icon(small=True, children=["mdi-vector-polygon"]),
                " Vector",
            ]

    def _on_map_interaction(self, **kwargs) -> None:
        lat, lon = kwargs.get("coordinates", (0, 0))
        self._update_status(lat=lat, lon=lon)

        if kwargs.get("type") != "click":
            return

        modifiers = kwargs.get("modifiers", {})
        if modifiers.get("ctrlKey"):
            webbrowser.open(f"https://www.google.com/maps/@{lat},{lon},18z", new=2)
            log_to_file("Handled as Ctrl-Click for Google Maps. Returning.")
            return

        if self.state.detection_mode:
            self._handle_detection_click(lon, lat)
            return

        if (
            not self.state.execute_label_point
            or self.state.lasso_mode
            or self.state.polygon_drawing
        ):
            return

        self.label_point(lon=lon, lat=lat)

    # ------------------------------------------------------------------
    # Labeling and drawing
    # ------------------------------------------------------------------

    def label_point(self, lon: float, lat: float) -> None:
        import time

        total_start = time.perf_counter()
        log_to_file("=" * 60)
        log_to_file(f"label_point: START (lon={lon:.4f}, lat={lat:.4f})")

        # Step 1: Query nearest point
        step_start = time.perf_counter()
        log_to_file("label_point: [1/4] Querying database for nearest point...")
        result = self.data.nearest_point(lon, lat)
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(f"label_point: [1/4] nearest_point completed in {step_time:.1f}ms")

        if result is None:
            self._show_operation_status("⚠️ No points found near click.")
            log_to_file("label_point: No points found, returning early")
            return

        # Step 2: Extract and cache embedding
        step_start = time.perf_counter()
        point_id = str(result[0])
        embedding = np.array(result[3], dtype=np.float32)
        self._cache_put_embeddings({point_id: embedding})
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(
            f"label_point: [2/4] Extract embedding completed in {step_time:.1f}ms (id={point_id})"
        )

        if self.state.select_val == UIConstants.ERASE_LABEL:
            erase_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings
            WHERE id = ?
            """
            erase_geojson = self.data.duckdb_connection.execute(
                erase_query, [point_id]
            ).fetchone()
            if erase_geojson:
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": json.loads(erase_geojson[0]),
                            "properties": {},
                        }
                    ],
                }
                self.map_manager.update_label_layers(
                    pos_geojson=self._empty_collection(),
                    neg_geojson=self._empty_collection(),
                    erase_geojson=geojson,
                )
            self.state.remove_label(point_id)
            self._show_operation_status("✅ Erased label")
        else:
            label_state = self.state.apply_label(point_id, self.state.select_val)
            status = "Positive" if label_state == "positive" else "Negative"
            if label_state == "removed":
                self._show_operation_status("✅ Removed label")
            else:
                self._show_operation_status(f"✅ Labeled point as {status}")

        # Step 3: Update map layers
        step_start = time.perf_counter()
        log_to_file("label_point: [3/4] Updating map layers...")
        self._update_layers()
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(f"label_point: [3/4] _update_layers completed in {step_time:.1f}ms")

        # Step 4: Update query vector
        step_start = time.perf_counter()
        log_to_file("label_point: [4/4] Updating query vector...")
        self._update_query_vector()
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(
            f"label_point: [4/4] _update_query_vector completed in {step_time:.1f}ms"
        )

        total_time = (time.perf_counter() - total_start) * 1000
        log_to_file(f"label_point: DONE total={total_time:.1f}ms")
        log_to_file("=" * 60)

    def _handle_detection_click(self, lon: float, lat: float) -> None:
        if not self.state.detection_data:
            return

        click_point = shapely.geometry.Point(lon, lat)
        features = self.state.detection_data.get("features", [])

        for feature in features:
            geom = shapely.geometry.shape(feature["geometry"])
            if geom.contains(click_point):
                props = feature.get("properties", {})
                tile_id = props.get("tile_id", props.get("id", "unknown"))
                probability = props.get("probability", 0.0)

                current_label = self.state.detection_labels.get(tile_id)
                if self.state.select_val == UIConstants.POSITIVE_LABEL:
                    new_label = 1
                    label_name = "positive (confirmed)"
                elif self.state.select_val == UIConstants.NEGATIVE_LABEL:
                    new_label = 0
                    label_name = "negative (hard negative)"
                else:
                    if tile_id in self.state.detection_labels:
                        del self.state.detection_labels[tile_id]
                        self._show_operation_status(
                            f"✅ Removed label from detection (P={probability:.2f})"
                        )
                    return

                if current_label == new_label:
                    del self.state.detection_labels[tile_id]
                    self._show_operation_status(
                        f"✅ Toggled off {label_name} (P={probability:.2f})"
                    )
                else:
                    self.state.label_detection(tile_id, new_label)
                    num_labeled = len(self.state.detection_labels)
                    self._show_operation_status(
                        f"✅ Marked as {label_name} (P={probability:.2f}) | "
                        f"Total labeled: {num_labeled}"
                    )
                self._refresh_detection_layer()
                return

        self._show_operation_status("⚠️ No detection at click location")

    def _handle_draw(self, target, action, geo_json) -> None:
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            polygon_coords = geo_json["geometry"]["coordinates"][0]
            polygon = shapely.geometry.Polygon(polygon_coords)

            # Detection mode: label detections within polygon
            if self.state.detection_mode and self.state.detection_data:
                self._label_detections_in_polygon(polygon)
                self.map_manager.draw_control.clear()
                return

            # Normal mode: label points within polygon
            point_ids: list[str] = []

            if self.state.detections_with_embeddings is not None:
                within_mask = self.state.detections_with_embeddings.geometry.within(
                    polygon
                )
                cached_points = self.state.detections_with_embeddings[within_mask]
                point_ids.extend(cached_points["id"].tolist())

            if not point_ids:
                polygon_wkt = polygon.wkt
                # Use local geometry cache if available (fast), otherwise remote
                if self.data._geometry_cache_connection is not None:
                    query = f"""
                    SELECT id
                    FROM geometry_cache
                    WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
                    """
                    conn = self.data._geometry_cache_connection
                else:
                    query = f"""
                    SELECT id
                    FROM geo_embeddings
                    WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
                    """
                    conn = self.data.duckdb_connection
                arrow_table = conn.execute(query).fetch_arrow_table()
                point_ids.extend(arrow_table.to_pandas()["id"].astype(str).tolist())

            if not point_ids:
                self._show_operation_status("⚠️ No points found within polygon")
                self.map_manager.draw_control.clear()
                self._update_status()
                return

            # Apply labels immediately (no embedding fetch needed)
            labeled = 0
            for pid in point_ids:
                result = self.state.apply_label(pid, self.state.select_val)
                if result != "removed":
                    labeled += 1

            self._update_layers()
            self.map_manager.draw_control.clear()

            # Check which embeddings need fetching
            all_labeled_ids = self.state.pos_ids + self.state.neg_ids
            uncached = self._filter_uncached_ids(all_labeled_ids)

            if uncached:
                # Start background prefetch, defer query vector update to search
                self._show_operation_status(
                    f"✅ Labeled {labeled} points | Loading embeddings..."
                )
                self._prefetch_embeddings_async(uncached, two_stage=True)
            else:
                # All cached, update query vector immediately
                self._update_query_vector()
                self._show_operation_status(
                    f"✅ Labeled {labeled} points as {self.state.current_label}"
                )
        elif action == "drawstart":
            self.state.polygon_drawing = True
            self._update_status()
        elif action == "deleted":
            self.state.polygon_drawing = False
            self._update_status()

    def _label_detections_in_polygon(self, polygon: shapely.geometry.Polygon) -> None:
        """Label all detections within the given polygon."""
        features = self.state.detection_data.get("features", [])
        labeled_count = 0

        # Determine label based on current mode
        if self.state.select_val == UIConstants.POSITIVE_LABEL:
            new_label = 1
            label_name = "positive"
        elif self.state.select_val == UIConstants.NEGATIVE_LABEL:
            new_label = 0
            label_name = "negative"
        else:
            # Erase mode: remove labels from detections in polygon
            for feature in features:
                geom = shapely.geometry.shape(feature["geometry"])
                if polygon.contains(geom.centroid) or polygon.intersects(geom):
                    props = feature.get("properties", {})
                    tile_id = props.get("tile_id", props.get("id", "unknown"))
                    if tile_id in self.state.detection_labels:
                        del self.state.detection_labels[tile_id]
                        labeled_count += 1
            self._show_operation_status(
                f"✅ Removed labels from {labeled_count} detections"
            )
            self._refresh_detection_layer()
            return

        for feature in features:
            geom = shapely.geometry.shape(feature["geometry"])
            # Check if detection centroid is within polygon or polygon intersects detection
            if polygon.contains(geom.centroid) or polygon.intersects(geom):
                props = feature.get("properties", {})
                tile_id = props.get("tile_id", props.get("id", "unknown"))
                self.state.label_detection(tile_id, new_label)
                labeled_count += 1

        total_labeled = len(self.state.detection_labels)
        self._show_operation_status(
            f"✅ Labeled {labeled_count} detections as {label_name} | Total: {total_labeled}"
        )
        self._refresh_detection_layer()

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------

    def search_click(self, _button=None) -> None:
        self.state.tile_page = 0
        self._reset_tiles_button()

        # If embeddings still loading, wait for them
        if self.state.embeddings_loading:
            self._show_operation_status("⏳ Waiting for embeddings to load...")
            self._wait_for_embeddings_then_search()
            return

        # Ensure we have embeddings for all labeled points
        all_labeled = self.state.pos_ids + self.state.neg_ids
        uncached = self._filter_uncached_ids(all_labeled)
        if uncached:
            self._show_operation_status(f"⏳ Fetching {len(uncached)} embeddings...")
            self._fetch_embeddings(uncached)
            self._update_query_vector()

        if self.state.query_vector is None or len(self.state.query_vector) == 0:
            if self.verbose:
                print("🔍 No query vector. Please label some points first.")
            self._show_operation_status("⚠️ Label some points to search")
            return
        self._search_faiss()

    def _wait_for_embeddings_then_search(self) -> None:
        """Wait for background embedding prefetch, then trigger search."""
        import threading
        import time

        def wait_and_search():
            # Poll until embeddings are ready (with timeout)
            timeout = 120  # 2 minutes max
            start = time.time()
            while self.state.embeddings_loading and (time.time() - start) < timeout:
                time.sleep(0.5)

            if self.state.embeddings_loading:
                log_to_file("_wait_for_embeddings: Timeout waiting for embeddings")
                return

            # Schedule search on main thread
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self.search_click)
            except RuntimeError:
                self.search_click()

        thread = threading.Thread(target=wait_and_search, daemon=True)
        thread.start()

    def _search_faiss(self) -> None:
        import time

        total_start = time.perf_counter()
        log_to_file("=" * 60)
        log_to_file("_search_faiss: START")

        n_neighbors = self.neighbors_slider.v_model
        all_labeled = self.state.pos_ids + self.state.neg_ids
        extra_results = min(len(all_labeled), n_neighbors // 2)
        total_requested = n_neighbors + extra_results
        log_to_file(
            f"_search_faiss: Requesting {total_requested} results (n_neighbors={n_neighbors})"
        )

        # Step 1: FAISS search
        step_start = time.perf_counter()
        log_to_file("_search_faiss: [1/3] Running FAISS search...")
        query_vector_np = self.state.query_vector.reshape(1, -1).astype("float32")
        if n_neighbors >= FAISS_NPROBE_HIGH_THRESHOLD:
            nprobe = FAISS_NPROBE_HIGH
        else:
            nprobe = FAISS_NPROBE_DEFAULT
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        log_to_file(f"_search_faiss: Using nprobe={nprobe} for n_neighbors={n_neighbors}")
        self._show_operation_status(
            f"🔍 FAISS Search: Finding {n_neighbors} neighbors..."
        )
        distances, ids = self.data.faiss_index.search(
            query_vector_np, total_requested, params=params
        )
        faiss_ids = ids[0].tolist()
        faiss_distances = distances[0].tolist()
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(
            f"_search_faiss: [1/3] FAISS search completed in {step_time:.1f}ms ({len(faiss_ids)} results)"
        )

        if not faiss_ids:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        # Step 2: Query metadata from DuckDB
        step_start = time.perf_counter()
        log_to_file(
            f"_search_faiss: [2/3] Querying metadata for {len(faiss_ids)} IDs..."
        )
        metadata_df = self.data.query_search_metadata(faiss_ids)
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(
            f"_search_faiss: [2/3] Metadata query completed in {step_time:.1f}ms"
        )

        if metadata_df is None or metadata_df.empty:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        id_map = {id_val: i for i, id_val in enumerate(faiss_ids)}
        metadata_df["sort_order"] = metadata_df["id"].map(id_map)
        metadata_df = metadata_df.sort_values("sort_order").drop(columns=["sort_order"])
        metadata_df["distance"] = faiss_distances[: len(metadata_df)]

        # Step 3: Process and display results
        step_start = time.perf_counter()
        log_to_file("_search_faiss: [3/3] Processing results...")
        self._process_search_results(metadata_df, n_neighbors)
        step_time = (time.perf_counter() - step_start) * 1000
        log_to_file(
            f"_search_faiss: [3/3] Process results completed in {step_time:.1f}ms"
        )

        total_time = (time.perf_counter() - total_start) * 1000
        log_to_file(f"_search_faiss: DONE total={total_time:.1f}ms")
        log_to_file("=" * 60)

        prefetch_top_k = self._suggest_prefetch_top_k(
            n_neighbors=n_neighbors,
            available_ids=len(faiss_ids),
        )
        if prefetch_top_k > 0:
            prefetch_ids = faiss_ids[:prefetch_top_k]
            log_to_file(
                "_search_faiss: adaptive prefetch "
                f"{prefetch_top_k}/{len(faiss_ids)} ids (n_neighbors={n_neighbors})"
            )
            self._prefetch_embeddings_async(prefetch_ids, two_stage=False)

    def _process_search_results(
        self, results_df: pd.DataFrame, n_neighbors: int
    ) -> None:
        all_labeled_ids = set(self.state.pos_ids + self.state.neg_ids)
        if not results_df.empty and all_labeled_ids:
            mask = ~results_df["id"].astype(str).isin(all_labeled_ids)
            filtered = results_df[mask].head(n_neighbors)
        else:
            filtered = results_df.head(n_neighbors)

        if filtered.empty:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        self._show_operation_status(f"✅ Found {len(filtered)} similar points.")

        geometries = [
            shapely.wkt.loads(row["geometry_wkt"]) for _, row in filtered.iterrows()
        ]
        display_column = getattr(self, "external_id_column", "id")
        base_columns = ["id", "distance"]
        if display_column != "id" and display_column in filtered.columns:
            base_columns.append(display_column)
        detections_df = filtered[base_columns].copy()
        detections_df["id"] = detections_df["id"].astype(str)
        if display_column in detections_df.columns:
            detections_df[display_column] = detections_df[display_column].astype(str)
        self.state.detections_with_embeddings = gpd.GeoDataFrame(
            detections_df,
            geometry=geometries,
            crs="EPSG:4326",
        )

        detections_geojson = {"type": "FeatureCollection", "features": []}
        min_distance = filtered["distance"].min()
        max_distance = filtered["distance"].max()
        highlight_cutoff = None
        if len(filtered) > 0:
            top_count = max(1, min(100, int(math.ceil(len(filtered) * 0.1))))
            highlight_cutoff = filtered.nsmallest(top_count, "distance")[
                "distance"
            ].max()
        for _, row in filtered.sort_values("distance", ascending=False).iterrows():
            color = UIConstants.distance_to_color(
                row["distance"], min_distance, max_distance, highlight_cutoff
            )
            display_id = self._display_id_from_row(row)
            props = {
                "id": str(row["id"]),
                "distance": row["distance"],
                "color": color,
                "fillColor": color,
                "source_id": display_id,
            }
            external_column_name = getattr(self, "external_id_column", "id")
            if external_column_name != "id" and external_column_name in row.index:
                props[external_column_name] = display_id
            detections_geojson["features"].append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry_json"]),
                    "properties": props,
                }
            )

        self.state.last_search_results_df = filtered.copy()
        self.map_manager.update_search_layer(
            detections_geojson,
            style_callback=self._search_style_callback,
        )
        self.tile_panel.update_results(
            filtered,
            auto_show=False,
            on_ready=self._on_tiles_ready,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mem_available_mb() -> Optional[float]:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split(":", 1)[1].strip().split()[0])
                        return kb / 1024.0
        except Exception:
            return None
        return None

    def _cache_contains_embedding(self, point_id: object, touch: bool = True) -> bool:
        key = str(point_id)
        cache = self.state.cached_embeddings
        if key not in cache:
            return False
        if touch and self._embedding_lru_size > 0:
            value = cache.pop(key)
            cache[key] = value
        return True

    def _trim_embedding_cache(self) -> None:
        if self._embedding_lru_size <= 0:
            return
        cache = self.state.cached_embeddings
        if len(cache) <= self._embedding_lru_size:
            return

        protected = {str(pid) for pid in (self.state.pos_ids + self.state.neg_ids)}
        target_size = self._embedding_lru_size
        removed = 0
        for key in list(cache.keys()):
            if len(cache) <= target_size:
                break
            if key in protected:
                continue
            cache.pop(key, None)
            removed += 1
        if removed > 0:
            log_to_file(
                f"embedding_lru: trimmed {removed} entries (size={len(cache)}, cap={target_size})"
            )

    def _cache_put_embeddings(self, embedding_map: Dict[str, np.ndarray]) -> None:
        if not embedding_map:
            return
        cache = self.state.cached_embeddings
        for key, value in embedding_map.items():
            skey = str(key)
            if skey in cache:
                cache.pop(skey)
            cache[skey] = np.asarray(value, dtype=np.float32)
        self._trim_embedding_cache()

    def _filter_uncached_ids(self, ids: list) -> list:
        uncached = []
        for point_id in ids:
            if not self._cache_contains_embedding(point_id):
                uncached.append(point_id)
        return uncached

    def _suggest_prefetch_top_k(self, n_neighbors: int, available_ids: int) -> int:
        target = int(round(float(n_neighbors) * PREFETCH_TOPK_RATIO))
        adaptive = max(PREFETCH_TOPK_MIN, min(PREFETCH_TOPK_MAX, target))
        return max(0, min(int(available_ids), adaptive))

    def _resolve_prefetch_workers(self, requested_workers: int, n_batches: int) -> int:
        if n_batches <= 0:
            return 1
        workers = max(1, requested_workers)
        workers = min(workers, PREFETCH_WORKER_CAP, n_batches)

        # Memory-aware cap keeps behavior safe on smaller VMs.
        mem_available_mb = self._mem_available_mb()
        mem_cap: Optional[int] = None
        if mem_available_mb is not None and PREFETCH_WORKER_MEM_MB > 0:
            mem_cap = max(1, int(mem_available_mb // PREFETCH_WORKER_MEM_MB))
            workers = min(workers, mem_cap)

        floor = min(PREFETCH_WORKER_FLOOR, n_batches)
        if mem_cap is not None:
            floor = min(floor, mem_cap)
        workers = max(1, min(workers, n_batches))
        if workers < floor:
            workers = floor
        return max(1, min(workers, n_batches))

    def _fetch_embeddings(self, point_ids):
        import time

        if not point_ids:
            return

        uncached_ids = self._filter_uncached_ids(point_ids)

        log_to_file(
            f"_fetch_embeddings: {len(point_ids)} requested, {len(uncached_ids)} uncached"
        )

        if not uncached_ids:
            log_to_file("_fetch_embeddings: 100% cache hit, skipping fetch")
            return

        start = time.perf_counter()
        embedding_map = self.data.fetch_embedding_map(uncached_ids)
        self._cache_put_embeddings(embedding_map)
        count = len(embedding_map)
        elapsed = (time.perf_counter() - start) * 1000
        log_to_file(f"_fetch_embeddings: Fetched {count} embeddings in {elapsed:.1f}ms")

    def _next_prefetch_generation(self) -> int:
        with self._prefetch_generation_lock:
            self._prefetch_generation += 1
            return self._prefetch_generation

    def _is_prefetch_generation_current(self, generation: int) -> bool:
        with self._prefetch_generation_lock:
            return generation == self._prefetch_generation

    def _prefetch_embeddings_async(
        self, ids: list, n_workers: int = PREFETCH_WORKER_TARGET, two_stage: bool = True
    ) -> None:
        """Pre-fetch embeddings using row-group-aware batching.

        Args:
            ids: Candidate IDs to prefetch.
            n_workers: Maximum worker threads.
            two_stage: If True, prefetch top-K first for faster readiness and
                continue tail in background. If False, fetch all in one stage.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ROW_GROUP_SIZE = 122880  # DuckDB default

        generation = self._next_prefetch_generation()
        uncached = self._filter_uncached_ids(ids)
        if not uncached:
            log_to_file(f"prefetch: All {len(ids)} embeddings already cached")
            self.state.embeddings_loading = False
            self.state.embeddings_pending_ids = []
            self._update_query_vector()
            return

        # Track loading state
        self.state.embeddings_loading = True
        self.state.embeddings_pending_ids = uncached

        if two_stage:
            fast_ids = uncached[: max(0, min(PREFETCH_FAST_K_DEFAULT, len(uncached)))]
            tail_ids = uncached[len(fast_ids) :]
        else:
            fast_ids = uncached
            tail_ids = []

        def fetch_worker():
            import time

            start_total = time.perf_counter()

            fast_batches, fast_group_mode, fast_lookup_ms = (
                self.data.group_ids_for_prefetch(fast_ids, row_group_size=ROW_GROUP_SIZE)
                if fast_ids
                else ([], "none", 0.0)
            )
            tail_batches, tail_group_mode, tail_lookup_ms = (
                self.data.group_ids_for_prefetch(tail_ids, row_group_size=ROW_GROUP_SIZE)
                if tail_ids
                else ([], "none", 0.0)
            )
            n_batches_total = len(fast_batches) + len(tail_batches)
            if n_batches_total == 0:
                if self._is_prefetch_generation_current(generation):
                    self.state.embeddings_loading = False
                    self.state.embeddings_pending_ids = []
                    self._on_prefetch_complete()
                return

            pool_workers = self._resolve_prefetch_workers(n_workers, n_batches_total)
            self.data.configure_background_connection_pool(pool_workers)
            pool_stats = self.data.background_pool_stats()
            log_to_file(
                f"prefetch[{generation}]: {len(uncached)} ids, "
                f"fast={len(fast_ids)} ({len(fast_batches)} batches, mode={fast_group_mode}, lookup={fast_lookup_ms:.1f}ms), "
                f"tail={len(tail_ids)} ({len(tail_batches)} batches, mode={tail_group_mode}, lookup={tail_lookup_ms:.1f}ms), "
                f"pool(size={pool_stats['size']}, created={pool_stats['created']}, idle={pool_stats['idle']})"
            )

            completed_batches = 0
            total_count = 0
            had_error = False
            fast_stage_elapsed_ms = 0.0
            tail_skipped_stale = False
            ready_notified = False

            def notify_ready_for_search(pending_ids: list[int]) -> None:
                nonlocal ready_notified
                if ready_notified:
                    return
                if not self._is_prefetch_generation_current(generation):
                    return
                ready_notified = True
                self.state.embeddings_loading = False
                self.state.embeddings_pending_ids = pending_ids
                try:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(self._on_prefetch_complete)
                except RuntimeError:
                    self._on_prefetch_complete()

            def run_stage(
                stage_name: str, stage_batches: list[list[int]], update_progress: bool = True
            ) -> tuple[int, float]:
                nonlocal completed_batches, had_error
                if not stage_batches:
                    return 0, 0.0

                stage_start = time.perf_counter()
                actual_workers = min(pool_workers, len(stage_batches))
                thread_local = threading.local()
                acquired_connections: list = []
                conn_lock = threading.Lock()

                def get_thread_connection():
                    conn = getattr(thread_local, "conn", None)
                    if conn is None:
                        conn = self.data.acquire_background_connection()
                        if conn is None:
                            raise RuntimeError("Failed to acquire background connection")
                        thread_local.conn = conn
                        with conn_lock:
                            acquired_connections.append(conn)
                    return conn

                def fetch_batch(batch_ids):
                    if not batch_ids:
                        return {}
                    conn = get_thread_connection()
                    return self.data.fetch_embedding_map_with_connection(conn, batch_ids)

                stage_count = 0
                try:
                    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                        futures = [executor.submit(fetch_batch, batch) for batch in stage_batches]
                        for future in as_completed(futures):
                            try:
                                chunk_result = future.result()
                            except Exception as exc:
                                had_error = True
                                log_to_file(
                                    f"prefetch[{generation}] {stage_name}: batch failed: {type(exc).__name__}: {exc}"
                                )
                                chunk_result = {}

                            self._cache_put_embeddings(chunk_result)
                            stage_count += len(chunk_result)
                            completed_batches += 1

                            if update_progress and (
                                completed_batches % 5 == 0
                                or completed_batches == n_batches_total
                            ):
                                pct = int(100 * completed_batches / n_batches_total)
                                try:
                                    import asyncio

                                    loop = asyncio.get_event_loop()
                                    loop.call_soon_threadsafe(
                                        lambda p=pct: self._show_operation_status(
                                            f"⏳ Loading embeddings... {p}%"
                                        )
                                    )
                                except RuntimeError:
                                    pass
                finally:
                    for conn in acquired_connections:
                        self.data.release_background_connection(conn)

                stage_elapsed_ms = (time.perf_counter() - stage_start) * 1000.0
                log_to_file(
                    f"prefetch[{generation}] {stage_name}: {stage_count} embeddings in {stage_elapsed_ms:.1f}ms"
                )
                return stage_count, stage_elapsed_ms

            try:
                stage_count, fast_stage_elapsed_ms = run_stage(
                    "fast", fast_batches, update_progress=True
                )
                total_count += stage_count
                notify_ready_for_search(tail_ids)

                if not self._is_prefetch_generation_current(generation):
                    tail_skipped_stale = True
                else:
                    stage_count, _ = run_stage("tail", tail_batches, update_progress=False)
                    total_count += stage_count

                elapsed = (time.perf_counter() - start_total) * 1000
                suffix = " (with batch errors)" if had_error else ""
                stale_suffix = " (tail skipped due newer prefetch)" if tail_skipped_stale else ""
                log_to_file(
                    f"prefetch[{generation}]: completed {total_count} embeddings "
                    f"in {elapsed/1000:.1f}s (fast_stage={fast_stage_elapsed_ms:.1f}ms){suffix}{stale_suffix}"
                )
            finally:
                # Only the latest prefetch run can finalize shared state.
                if not self._is_prefetch_generation_current(generation):
                    log_to_file(f"prefetch[{generation}]: stale completion ignored")
                else:
                    if not ready_notified:
                        notify_ready_for_search([])
                    else:
                        self.state.embeddings_loading = False
                        self.state.embeddings_pending_ids = []

        thread = threading.Thread(target=fetch_worker, daemon=True)
        thread.start()

    def _on_prefetch_complete(self) -> None:
        """Called when background prefetch completes."""
        self._update_query_vector()
        n_pos = len(self.state.pos_ids)
        n_neg = len(self.state.neg_ids)
        self._show_operation_status(f"✅ Ready to search ({n_pos}+ / {n_neg}-)")

    def _update_layers(self) -> None:
        pos_geojson = self._geojson_for_ids(self.state.pos_ids)
        neg_geojson = self._geojson_for_ids(self.state.neg_ids)
        self.map_manager.update_label_layers(
            pos_geojson=pos_geojson,
            neg_geojson=neg_geojson,
            erase_geojson=self._empty_collection(),
        )

    def _geojson_for_ids(self, ids):
        if not ids:
            return self._empty_collection()
        prepared_ids = [str(pid) for pid in ids]
        placeholders = ",".join(["?" for _ in prepared_ids])

        # Use local geometry cache if available (fast), otherwise fall back to remote
        if self.data._geometry_cache_connection is not None:
            query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geometry_cache
            WHERE id IN ({placeholders})
            """
            df = self.data._geometry_cache_connection.execute(query, prepared_ids).df()
        else:
            query = f"""
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
            df = self.data.duckdb_connection.execute(query, prepared_ids).df()

        features = [
            {
                "type": "Feature",
                "geometry": json.loads(row["geometry"]),
                "properties": {},
            }
            for _, row in df.iterrows()
        ]
        return {"type": "FeatureCollection", "features": features}

    def _update_query_vector(self) -> None:
        if not self.state.pos_ids:
            self.state.query_vector = None
            return
        self._fetch_embeddings(self.state.pos_ids)
        if self.state.neg_ids:
            self._fetch_embeddings(self.state.neg_ids)
        self.state.update_query_vector()

    def _on_tiles_ready(self) -> None:
        self.tiles_button.color = "success"
        self.tiles_button.outlined = False

    def _reset_tiles_button(self) -> None:
        self.tiles_button.color = None
        self.tiles_button.outlined = True

    def _display_id_from_row(self, row) -> str:
        column = getattr(self, "external_id_column", "id")
        if column != "id" and column in row.index:
            value = row[column]
            if pd.isna(value):
                return str(row["id"])
            return str(value)
        for candidate in ("source_id", "tile_id"):
            if candidate in row.index and not pd.isna(row[candidate]):
                return str(row[candidate])
        return str(row["id"])

    def _handle_tile_label(self, point_id: str, row, label: str) -> None:
        # Detection mode: label detections differently
        if self.state.detection_mode:
            tile_id = point_id  # In detection mode, point_id is the tile_id

            if label == UIConstants.POSITIVE_LABEL:
                new_label = 1
                label_name = "positive"
            else:
                new_label = 0
                label_name = "negative"

            current_label = self.state.detection_labels.get(tile_id)
            if current_label == new_label:
                # Toggle off
                del self.state.detection_labels[tile_id]
                self._show_operation_status("✅ Removed label from detection")
            else:
                self.state.label_detection(tile_id, new_label)
                num_labeled = len(self.state.detection_labels)
                self._show_operation_status(
                    f"✅ Labeled as {label_name} | Total: {num_labeled}"
                )
            self._refresh_detection_layer()
            return

        # Normal mode: fetch embeddings and apply label
        if not self._cache_contains_embedding(point_id):
            self._fetch_embeddings([point_id])
        result = self.state.apply_label(point_id, label)
        if result == "positive":
            self._show_operation_status("✅ Labeled tile as Positive")
        elif result == "negative":
            self._show_operation_status("✅ Labeled tile as Negative")
        else:
            self._show_operation_status("✅ Removed label from tile")
        self._update_layers()
        self._update_query_vector()

    def _handle_tile_center(self, row) -> None:
        geom = shapely.wkt.loads(row["geometry_wkt"])
        lat, lon = geom.y, geom.x
        self.map_manager.center_on(lat, lon, zoom=14)

        polygon = self._tile_polygon_from_spec(lat, lon)
        if polygon is None:
            half_size = 0.0025 / 2
            square_coords = [
                (lon - half_size, lat - half_size),
                (lon + half_size, lat - half_size),
                (lon + half_size, lat + half_size),
                (lon - half_size, lat + half_size),
                (lon - half_size, lat - half_size),
            ]
            polygon = shapely.geometry.Polygon(square_coords)

        self.map_manager.highlight_polygon(polygon, color="red", fill_opacity=0.0)
        self._show_operation_status("📍 Centered on tile")

    def _tile_polygon_from_spec(self, lat: float, lon: float):
        tile_spec = getattr(self.data, "tile_spec", None)
        if not tile_spec:
            return None

        meters_per_pixel = tile_spec.get("meters_per_pixel")
        tile_size_px = tile_spec.get("tile_size_px")
        if not meters_per_pixel or not tile_size_px:
            return None

        half_side = (meters_per_pixel * tile_size_px) / 2.0
        if half_side <= 0:
            return None

        zone = int((lon + 180) // 6) + 1
        zone = max(1, min(zone, 60))
        epsg = 32600 + zone if lat >= 0 else 32700 + zone

        try:
            utm_crs = pyproj.CRS.from_epsg(epsg)
            forward = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            inverse = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
            x, y = forward.transform(lon, lat)
            square = shapely.geometry.box(
                x - half_side,
                y - half_side,
                x + half_side,
                y + half_side,
            )
            return shapely.ops.transform(inverse.transform, square)
        except Exception:
            return None

    def _filter_detection_layer(self, threshold: float) -> None:
        if not self.state.detection_data:
            return
        features = self.state.detection_data.get("features", [])
        filtered_features = [
            f
            for f in features
            if f.get("properties", {}).get("probability", 0.0) >= threshold
        ]
        filtered_geojson = {"type": "FeatureCollection", "features": filtered_features}
        self.map_manager.update_detection_layer(
            filtered_geojson, style_callback=self._detection_style_callback
        )
        num_shown = len(filtered_features)
        num_total = len(features)
        num_labeled = len(self.state.detection_labels)
        self._show_operation_status(
            f"🔍 {num_shown}/{num_total} detections | {num_labeled} labeled"
        )

    def _refresh_detection_layer(self) -> None:
        """Refresh detection layer with current threshold and labels."""
        threshold = self.detection_threshold_slider.v_model
        self._filter_detection_layer(threshold)

    def _update_detection_tiles(self) -> None:
        """Update the tile panel with current detections, sorted by lowest probability first."""
        if not self.state.detection_mode or not self.state.detection_data:
            return

        threshold = self.detection_threshold_slider.v_model
        features = self.state.detection_data.get("features", [])

        # Filter by threshold
        filtered = [
            f
            for f in features
            if f.get("properties", {}).get("probability", 0.0) >= threshold
        ]

        if not filtered:
            self.tile_panel.clear()
            return

        # Build DataFrame for tile panel (sorted by lowest probability first)
        records = []
        for feature in filtered:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            tile_id = props.get("tile_id", props.get("id", "unknown"))
            probability = props.get("probability", 0.5)

            # Convert geometry to WKT
            geom_shape = shapely.geometry.shape(geom)
            centroid = geom_shape.centroid

            records.append(
                {
                    "id": str(tile_id),
                    "probability": probability,
                    "geometry_wkt": centroid.wkt,
                    "geometry_json": json.dumps(shapely.geometry.mapping(centroid)),
                }
            )

        df = pd.DataFrame(records)
        # Sort by probability ascending (lowest first = hardest cases)
        df = df.sort_values("probability", ascending=True).reset_index(drop=True)

        # Update tile panel
        self.tile_panel.update_results(df, auto_show=False)

    def _detection_style_callback(self, feature):
        """Style callback for detection layer that shows labeled detections in pos/neg colors."""
        from geovibes.ui_config import LayerStyles

        props = feature.get("properties", {})
        tile_id = props.get("tile_id", props.get("id", "unknown"))
        probability = props.get("probability", 0.5)

        # Check if this detection has been labeled
        label = self.state.detection_labels.get(tile_id)

        if label == 1:
            # Labeled as positive - use blue
            color = UIConstants.POS_COLOR
        elif label == 0:
            # Labeled as negative - use orange
            color = UIConstants.NEG_COLOR
        else:
            # Not labeled - normalize probability to dataset range for colormap
            min_prob = getattr(self, "_detection_prob_min", 0.0)
            max_prob = getattr(self, "_detection_prob_max", 1.0)
            if max_prob > min_prob:
                normalized = (probability - min_prob) / (max_prob - min_prob)
            else:
                normalized = 0.5
            color = LayerStyles.probability_to_color(normalized)

        return {
            "color": color,
            "weight": 3 if label is not None else 2,
            "opacity": 0.9 if label is not None else 0.8,
            "fillColor": color,
            "fillOpacity": 0.2 if label is not None else 0.1,
        }

    def _handle_save_dataset(self) -> None:
        if self.state.detection_mode:
            result = self.dataset_manager.export_augmented_dataset()
        else:
            result = self.dataset_manager.save_dataset()

        if result:
            geojson_path = result.get("geojson")
            csv_path = result.get("csv")
            if geojson_path and csv_path:
                message = f"✅ Dataset saved: {geojson_path} (labels: {csv_path})"
            elif geojson_path:
                message = f"✅ Dataset saved: {geojson_path}"
            else:
                message = "✅ Dataset saved"
            self._show_operation_status(message)
        else:
            self._show_operation_status("⚠️ Nothing to save")

    def reset_all(self, _button=None, clear_overlays: bool = False) -> None:
        if self.verbose:
            print("🗑️ Resetting all labels and search results...")
        self.state.reset()
        self.map_manager.update_label_layers(
            pos_geojson=self._empty_collection(),
            neg_geojson=self._empty_collection(),
            erase_geojson=self._empty_collection(),
        )
        self.map_manager.update_search_layer(self._empty_collection())
        self.map_manager.clear_detection_layer()
        self.map_manager.clear_vector_layer()
        self.map_manager.clear_highlight()
        if clear_overlays:
            self.map_manager.clear_overlay_layers()
        self.detection_controls.style_ = "display: none;"
        # Reset slider and colormap range to defaults
        self.detection_threshold_slider.min = 0.0
        self.detection_threshold_slider.max = 1.0
        self.detection_threshold_slider.v_model = 0.5
        self.detection_threshold_label.children = ["0.50"]
        self._detection_prob_min = 0.0
        self._detection_prob_max = 1.0
        self.tile_panel.clear()
        self.tile_panel.hide()
        self._clear_operation_status()
        self._update_status()

    def _search_style_callback(self, feature):
        props = feature.get("properties", {})
        return {
            "color": "black",
            "radius": UIConstants.SEARCH_POINT_RADIUS,
            "fillColor": props.get("fillColor", UIConstants.SEARCH_COLOR),
            "opacity": UIConstants.POINT_OPACITY,
            "fillOpacity": UIConstants.POINT_FILL_OPACITY,
            "weight": UIConstants.SEARCH_POINT_WEIGHT,
        }

    def _update_status(
        self, lat: Optional[float] = None, lon: Optional[float] = None
    ) -> None:
        self.map_manager.update_status(lat=lat, lon=lon)

    def _show_operation_status(self, message: str) -> None:
        self.map_manager.set_operation(message)

    def _clear_operation_status(self) -> None:
        self.map_manager.clear_operation()

    @staticmethod
    def _empty_collection() -> Dict:
        return {"type": "FeatureCollection", "features": []}

    # ------------------------------------------------------------------
    # Overlay tile layer API
    # ------------------------------------------------------------------

    def add_tile_layer(
        self, url: str, name: str, opacity: float = 1.0, attribution: str = ""
    ) -> None:
        """Add an XYZ tile layer overlay."""
        self.map_manager.add_tile_layer(url, name, opacity, attribution)

    def add_ee_layer(
        self, ee_image, vis_params: Dict, name: str, opacity: float = 1.0
    ) -> None:
        """Add an Earth Engine image as a tile layer overlay."""
        self.map_manager.add_ee_layer(ee_image, vis_params, name, opacity)

    def remove_layer(self, name: str) -> bool:
        """Remove an overlay layer by name."""
        return self.map_manager.remove_layer(name)

    def set_layer_opacity(self, name: str, opacity: float) -> None:
        """Set the opacity of an overlay layer."""
        self.map_manager.set_layer_opacity(name, opacity)

    def list_layers(self) -> List[str]:
        """Return names of all overlay layers."""
        return self.map_manager.list_overlay_layers()

    def close(self) -> None:
        self.data.close()


__all__ = ["GeoVibes"]
