"""Tile panel for displaying search results as thumbnails."""

from __future__ import annotations

import asyncio
import contextvars
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

import ipywidgets as ipyw
import shapely.wkt
from ipywidgets import Button, GridBox, HBox, HTML, Image, Label, Layout, VBox

from geovibes.ui_config import BasemapConfig, UIConstants
from .xyz import get_map_image

TILE_SOURCES = ("HUTCH_TILE", "MAPTILER", "GOOGLE_HYBRID")

TILE_PANEL_CSS = """
<style>
/* Container */
.tile-panel-container {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15), 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.8);
}

/* Tile Cards */
.tile-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden !important;
    border: 2px solid transparent;
}
.tile-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    transform: translateY(-2px);
    border-color: #3b82f6;
}

/* Labeled tile states */
.tile-positive {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 1px #22c55e, 0 2px 8px rgba(34,197,94,0.25) !important;
}
.tile-negative {
    border-color: #ef4444 !important;
    box-shadow: 0 0 0 1px #ef4444, 0 2px 8px rgba(239,68,68,0.25) !important;
}

/* Header */
.tile-header {
    background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
    border-bottom: 1px solid #e2e8f0;
    padding: 8px 10px;
    border-radius: 12px 12px 0 0;
}

/* Styled dropdown */
.styled-dropdown select,
.styled-dropdown .widget-dropdown-input {
    background: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    padding: 2px 24px 2px 8px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    height: 26px !important;
    line-height: 22px !important;
}
.styled-dropdown select:hover {
    border-color: #3b82f6 !important;
    background: #f8fafc !important;
}
.styled-dropdown select:focus {
    outline: none !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Footer */
.tile-footer {
    background: linear-gradient(180deg, #fafbfc 0%, #ffffff 100%);
    border-top: 1px solid #e2e8f0;
    padding: 8px 10px;
    border-radius: 0 0 12px 12px;
}

/* Load More Button */
.load-more-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(59,130,246,0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}
.load-more-btn:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
    transform: translateY(-1px);
}

/* Page info */
.page-info-text {
    color: #64748b;
    font-size: 11px;
    font-weight: 500;
}

/* Sort Toggle Buttons - active state (blue) */
.sort-btn-active {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(59,130,246,0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Sort Toggle Buttons - inactive state (white) */
.sort-btn-inactive {
    background: white !important;
    color: #6b7280 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.sort-btn-inactive:hover {
    background: #f8fafc !important;
    border-color: #3b82f6 !important;
    color: #374151 !important;
}

/* Action buttons - no scrollbar */
.tile-action-btn {
    opacity: 0.5;
    transition: all 0.15s ease !important;
    border-radius: 4px !important;
    border: none !important;
    background: transparent !important;
}
.tile-action-btn:hover {
    opacity: 1;
    background: rgba(59,130,246,0.1) !important;
}

/* Scrollbar styling */
.tile-scroll-area::-webkit-scrollbar {
    width: 5px;
}
.tile-scroll-area::-webkit-scrollbar-track {
    background: transparent;
}
.tile-scroll-area::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}
.tile-scroll-area::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* Hide any unwanted scrollbars */
.tile-card * {
    overflow: hidden !important;
}
.tile-panel-container {
    overflow: hidden !important;
}
.tile-scroll-area {
    overflow-x: hidden !important;
}
</style>
"""


class TilePanel:
    """Manages the tile-based results pane."""

    def __init__(
        self,
        *,
        state,
        map_manager,
        on_label: Callable[[str, dict, str], None],
        on_center: Callable[[dict], None],
        verbose: bool = False,
    ) -> None:
        self.state = state
        self.map_manager = map_manager
        self.on_label = on_label
        self.on_center = on_center
        self.verbose = verbose

        self.allowed_sources = [
            name for name in TILE_SOURCES if name in BasemapConfig.BASEMAP_TILES
        ]
        if not self.allowed_sources:
            self.allowed_sources = list(BasemapConfig.BASEMAP_TILES.keys())

        if self.state.tile_basemap not in self.allowed_sources:
            self.state.tile_basemap = self.allowed_sources[0]

        self._css_widget = HTML(TILE_PANEL_CSS)

        self.tile_basemap_dropdown = ipyw.Dropdown(
            options=self.allowed_sources,
            value=self.state.tile_basemap,
            description="",
            layout=Layout(width="120px", height="28px"),
            style={"description_width": "initial"},
        )
        self.tile_basemap_dropdown.add_class("styled-dropdown")
        self.tile_basemap_dropdown.observe(self._on_tile_basemap_change, names="value")

        self.page_info_label = HTML(
            value="",
            layout=Layout(margin="0 8px 0 auto"),
        )
        self.page_info_label.add_class("page-info-text")

        self._sort_order = "Similar"

        self.similar_btn = Button(
            description="Similar",
            layout=Layout(width="50%", height="28px"),
        )
        self.similar_btn.add_class("sort-btn-active")
        self.similar_btn.on_click(self._on_similar_click)

        self.dissimilar_btn = Button(
            description="Dissimilar",
            layout=Layout(width="50%", height="28px"),
        )
        self.dissimilar_btn.add_class("sort-btn-inactive")
        self.dissimilar_btn.on_click(self._on_dissimilar_click)

        self.geo_filter_btn = Button(
            description="📍 Filter Area",
            layout=Layout(width="100%", height="28px"),
        )
        self.geo_filter_btn.add_class("sort-btn-inactive")
        self.geo_filter_btn.on_click(self._on_geo_filter_click)

        self.load_more_btn = Button(
            description="Load More",
            layout=Layout(
                width="100%",
                height="32px",
                display="flex",
                justify_content="center",
            ),
        )
        self.load_more_btn.add_class("load-more-btn")
        self.load_more_btn.on_click(self._on_next_tiles_click)

        self.next_tiles_btn = self.load_more_btn

        header_row = HBox(
            [self.tile_basemap_dropdown, self.page_info_label],
            layout=Layout(
                align_items="center",
                padding="8px 8px 4px 8px",
                width="100%",
            ),
        )

        # Sort toggle row: two buttons side by side
        sort_row = HBox(
            [self.similar_btn, self.dissimilar_btn],
            layout=Layout(
                padding="0 8px 8px 8px",
                width="100%",
                gap="6px",
            ),
        )

        # Geographic filter row: full width filter toggle
        geo_filter_row = HBox(
            [self.geo_filter_btn],
            layout=Layout(
                padding="0 8px 8px 8px",
                width="100%",
            ),
        )

        # Load More button row: full width prominent button
        load_more_row = HBox(
            [self.load_more_btn],
            layout=Layout(
                padding="0 8px 8px 8px",
                width="100%",
            ),
        )

        header = VBox(
            [header_row, sort_row, geo_filter_row, load_more_row],
            layout=Layout(width="100%"),
        )
        header.add_class("tile-header")

        self.results_grid = GridBox(
            [],
            layout=Layout(
                width="100%",
                grid_template_columns="1fr 1fr",
                grid_gap="8px",
                padding="8px",
            ),
        )

        # Fixed height for scroll area: 4 rows of tiles
        tile_height = 148
        grid_gap = 8
        visible_rows = 4
        scroll_height = (tile_height + grid_gap) * visible_rows

        self.scroll_area = VBox(
            [self.results_grid],
            layout=Layout(
                width="100%",
                height=f"{scroll_height}px",
                min_height=f"{scroll_height}px",
                max_height=f"{scroll_height}px",
                overflow_y="auto",
                overflow_x="hidden",
            ),
        )
        self.scroll_area.add_class("tile-scroll-area")

        self.container = VBox(
            [self._css_widget, header, self.scroll_area],
            layout=Layout(
                display="none",
                width="270px",
                padding="0px",
                border_radius="8px",
                overflow_x="hidden",
                overflow_y="hidden",
            ),
        )
        self.container.add_class("tile-panel-container")

        self.control = self.map_manager.add_widget_control(
            self.container, position="topright"
        )

        self.results_ready = False
        self._tiles_ready_callback: Optional[Callable[[], None]] = None
        self._page_sizes: List[int] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self._pending_futures: set[concurrent.futures.Future] = set()
        self._pending_batches: Dict[object, Dict[str, Any]] = {}
        self._loader_token: Optional[object] = None
        try:
            self._async_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._async_loop = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self.container.layout.display == "none":
            self.container.layout.display = "flex"
            if (
                not self.results_ready
                and self.state.last_search_results_df is not None
                and not self.results_grid.children
            ):
                self._render_current_page(
                    append=False,
                    on_finish=self._handle_tiles_ready,
                    loader_token=self._loader_token,
                )
        else:
            self.container.layout.display = "none"

    def show(self) -> None:
        self.container.layout.display = "flex"

    def hide(self) -> None:
        self.container.layout.display = "none"

    def clear(self) -> None:
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self.results_grid.children = []
        self.next_tiles_btn.layout.display = "none"
        self.page_info_label.value = ""
        self.state.last_search_results_df = None
        self.state.tile_page = 0
        self._update_operation(None)
        self.results_ready = False
        self._tiles_ready_callback = None
        self._page_sizes = []
        self._loader_token = None
        if self.state.geo_filter_mode:
            self.state.geo_filter_mode = False
            self.state.geo_filter_polygon = None
            self.geo_filter_btn.remove_class("sort-btn-active")
            self.geo_filter_btn.add_class("sort-btn-inactive")
            self.geo_filter_btn.description = "📍 Filter Area"

    def update_results(
        self,
        search_results_df,
        *,
        auto_show: bool = False,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self._tiles_ready_callback = on_ready
        self.results_ready = False
        self._page_sizes = []
        self.state.last_search_results_df = search_results_df
        self.state.tile_page = 0
        self.results_grid.children = []
        self.next_tiles_btn.layout.display = "none"
        token = object()
        self._loader_token = token
        if auto_show:
            self.show()
        self._render_current_page(
            append=False, on_finish=self._handle_tiles_ready, loader_token=token
        )

    def reload_tiles_for_new_basemap(self) -> None:
        if self.state.last_search_results_df is None:
            return
        current_count = len(self.results_grid.children)
        if current_count == 0:
            return
        self._render_current_page(
            append=False,
            limit=current_count,
            loader_token=self._loader_token,
        )

    def apply_geographic_filter(self) -> None:
        if self.state.last_search_results_df is None:
            return
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self.state.tile_page = 0
        self._page_sizes = []
        self.results_grid.children = []
        self._render_current_page(
            append=False,
            on_finish=self._handle_tiles_ready,
            loader_token=self._loader_token,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_tile_basemap_change(self, change) -> None:
        new_value = change["new"]
        if new_value not in self.allowed_sources:
            return
        self.state.tile_basemap = new_value
        self.reload_tiles_for_new_basemap()

    def _on_similar_click(self, _button) -> None:
        if self._sort_order == "Similar":
            return
        self._sort_order = "Similar"
        self.similar_btn.remove_class("sort-btn-inactive")
        self.similar_btn.add_class("sort-btn-active")
        self.dissimilar_btn.remove_class("sort-btn-active")
        self.dissimilar_btn.add_class("sort-btn-inactive")
        self._trigger_sort_refresh()

    def _on_dissimilar_click(self, _button) -> None:
        if self._sort_order == "Dissimilar":
            return
        self._sort_order = "Dissimilar"
        self.dissimilar_btn.remove_class("sort-btn-inactive")
        self.dissimilar_btn.add_class("sort-btn-active")
        self.similar_btn.remove_class("sort-btn-active")
        self.similar_btn.add_class("sort-btn-inactive")
        self._trigger_sort_refresh()

    def _on_geo_filter_click(self, _button) -> None:
        self.state.geo_filter_mode = not self.state.geo_filter_mode
        if self.state.geo_filter_mode:
            self.state.execute_label_point = False
            self.state.lasso_mode = False
            self.geo_filter_btn.remove_class("sort-btn-inactive")
            self.geo_filter_btn.add_class("sort-btn-active")
            self.geo_filter_btn.description = "📍 Filter Active"
            self._update_operation("📍 Draw polygon to filter tiles by area")
        else:
            self.state.execute_label_point = True
            self.geo_filter_btn.remove_class("sort-btn-active")
            self.geo_filter_btn.add_class("sort-btn-inactive")
            self.geo_filter_btn.description = "📍 Filter Area"
            self.state.geo_filter_polygon = None
            self._trigger_sort_refresh()
            self._update_operation(None)

    def _trigger_sort_refresh(self) -> None:
        if self.state.last_search_results_df is None:
            return
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self.state.tile_page = 0
        self._page_sizes = []
        self.results_grid.children = []
        self._render_current_page(
            append=False,
            on_finish=self._handle_tiles_ready,
            loader_token=self._loader_token,
        )

    def _on_next_tiles_click(self, _button) -> None:
        next_size = 10
        self.state.tile_page += 1
        self._render_current_page(
            append=True,
            page_size=next_size,
            loader_token=self._loader_token,
        )

    def handle_map_basemap_change(self, basemap_name: str) -> None:
        if basemap_name in self.allowed_sources:
            if self.tile_basemap_dropdown.value != basemap_name:
                self.tile_basemap_dropdown.value = basemap_name
            else:
                self.state.tile_basemap = basemap_name
                self.reload_tiles_for_new_basemap()
        elif self.tile_basemap_dropdown.value not in self.allowed_sources:
            self.tile_basemap_dropdown.value = self.allowed_sources[0]

    def _render_current_page(
        self,
        append: bool,
        limit: Optional[int] = None,
        on_finish: Optional[Callable[[], None]] = None,
        page_size: Optional[int] = None,
        loader_token: Optional[object] = None,
    ) -> None:
        df = self.state.last_search_results_df
        if df is None or df.empty:
            self.results_grid.children = []
            self.next_tiles_btn.layout.display = "none"
            self._update_operation(None)
            if on_finish:
                on_finish()
            return

        # Apply geographic filter if active
        if (
            self.state.geo_filter_mode
            and self.state.geo_filter_polygon is not None
            and "geometry_wkt" in df.columns
        ):
            filter_polygon = self.state.geo_filter_polygon
            mask = df["geometry_wkt"].apply(
                lambda wkt: shapely.wkt.loads(wkt).within(filter_polygon)
            )
            df = df[mask].reset_index(drop=True)
            if df.empty:
                self.results_grid.children = []
                self.next_tiles_btn.layout.display = "none"
                self._update_operation("⚠️ No tiles within filter area")
                if on_finish:
                    on_finish()
                return

        # Determine if we should reverse the order
        # - Search mode: data is sorted by distance ascending (low = similar)
        #   "Similar" = keep order, "Dissimilar" = reverse
        # - Detection mode: data is sorted by probability ascending (low = dissimilar)
        #   "Similar" = reverse (high prob first), "Dissimilar" = keep order
        is_detection = getattr(self.state, "detection_mode", False)
        should_reverse = (self._sort_order == "Dissimilar") != is_detection
        if should_reverse:
            df = df.iloc[::-1].reset_index(drop=True)

        refresh_only = limit is not None and not append

        if refresh_only:
            start_index = 0
            end_index = min(limit, len(df))
        else:
            already_loaded = sum(self._page_sizes)
            default_size = (
                self.state.tiles_per_page
                if self._page_sizes
                else self.state.initial_load_size
            )

            desired = page_size if page_size is not None else default_size
            desired = max(1, desired)

            start_index = already_loaded if append or self._page_sizes else 0
            end_index = min(start_index + desired, len(df))

        page_df = df.iloc[start_index:end_index]
        if page_df.empty:
            self.next_tiles_btn.layout.display = "none"
            if on_finish:
                on_finish()
            return

        placeholder_widgets = [
            self._make_placeholder_tile() for _ in range(len(page_df))
        ]
        current_children = list(self.results_grid.children)
        if append:
            placeholder_start = len(current_children)
            current_children.extend(placeholder_widgets)
        else:
            placeholder_start = 0
            current_children = placeholder_widgets
        self.results_grid.children = tuple(current_children)

        placeholder_indices = [
            placeholder_start + idx for idx in range(len(placeholder_widgets))
        ]

        batch_token = loader_token if loader_token is not None else object()
        remaining = len(page_df)
        batch_info = {
            "remaining": remaining,
            "on_finish": on_finish if not append else None,
            "token": loader_token,
            "update_status": "✅ Tiles updated",
        }
        self._pending_batches[batch_token] = batch_info

        if not refresh_only:
            if not append:
                self._page_sizes = [remaining]
            else:
                self._page_sizes.append(remaining)

        self._update_operation("⏳ Loading tiles...")

        for offset, (_, row) in enumerate(page_df.iterrows()):
            target_index = placeholder_indices[offset]
            row_data = row
            ctx = contextvars.copy_context()
            future = self._executor.submit(self._fetch_tile_image_bytes, row_data)
            self._pending_futures.add(future)
            future.add_done_callback(
                lambda fut,
                idx=target_index,
                bt=batch_token,
                r=row_data,
                ctx=ctx: self._on_tile_future_done(fut, idx, bt, r, ctx)
            )

        if end_index < len(df):
            self.next_tiles_btn.layout.display = "flex"
        else:
            self.next_tiles_btn.layout.display = "none"

        self._update_page_info(end_index, len(df))

    def _handle_tiles_ready(self) -> None:
        if self.results_ready:
            return
        self.results_ready = True
        callback = self._tiles_ready_callback
        self._tiles_ready_callback = None
        if callback:
            callback()

    def _dispatch_to_ui(
        self,
        func: Callable[..., None],
        *args,
        context: Optional[contextvars.Context] = None,
    ) -> None:
        if context is not None:

            def runner() -> None:
                context.run(func, *args)
        else:

            def runner() -> None:
                func(*args)

        if self._async_loop and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(runner)
        else:
            runner()

    def _on_tile_future_done(
        self,
        future: concurrent.futures.Future,
        target_index: int,
        batch_token: object,
        row,
        context: contextvars.Context,
    ) -> None:
        self._pending_futures.discard(future)
        batch = self._pending_batches.get(batch_token)
        if batch is None:
            return
        token = batch.get("token")
        if future.cancelled():
            self._pending_batches.pop(batch_token, None)
            return
        try:
            image_bytes = future.result()
        except Exception:
            image_bytes = None

        def apply_result() -> None:
            current_batch = self._pending_batches.get(batch_token)
            if current_batch is None:
                return
            if token is not None and token is not self._loader_token:
                self._pending_batches.pop(batch_token, None)
                return
            current_children = list(self.results_grid.children)
            if target_index < len(current_children):
                try:
                    rank = target_index + 1
                    widget = self._build_tile_widget(row, image_bytes, rank=rank)
                except Exception:
                    widget = self._make_placeholder_tile()
                current_children[target_index] = widget
                self.results_grid.children = tuple(current_children)
            current_batch["remaining"] -= 1
            if current_batch["remaining"] <= 0:
                self._pending_batches.pop(batch_token, None)
                if current_batch.get("update_status"):
                    self._update_operation(current_batch["update_status"])
                finish_cb = current_batch.get("on_finish")
                if finish_cb and (token is None or token is self._loader_token):
                    finish_cb()

        self._dispatch_to_ui(apply_result, context=context)

    def _fetch_tile_image_bytes(self, row) -> Optional[bytes]:
        try:
            geom = shapely.wkt.loads(row["geometry_wkt"])
        except Exception:
            if self.verbose:
                print(f"Failed to parse geometry for tile {row.get('id')}")
            return None

        tile_spec = getattr(getattr(self.map_manager, "data", None), "tile_spec", None)
        try:
            return get_map_image(
                source=self.state.tile_basemap,
                lon=geom.x,
                lat=geom.y,
                tile_spec=tile_spec,
            )
        except Exception:
            if self.verbose:
                print(f"Failed to fetch tile image for {row.get('id')}")
            return None

    def _build_tile_widget(
        self, row, image_bytes: Optional[bytes], rank: Optional[int] = None
    ) -> VBox:
        img_size = 116
        point_id = str(row["id"])

        if image_bytes:
            tile_image = Image(
                value=image_bytes,
                format="png",
                width=img_size,
                height=img_size,
                layout=Layout(
                    width=f"{img_size}px",
                    height=f"{img_size}px",
                    overflow="hidden",
                    border_radius="6px 6px 0 0",
                ),
            )
        else:
            tile_image = Label(
                value="No image",
                layout=Layout(
                    width=f"{img_size}px",
                    height=f"{img_size}px",
                    overflow="hidden",
                    background="#f1f5f9",
                    border_radius="6px 6px 0 0",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    color="#94a3b8",
                    font_size="11px",
                ),
            )

        image_container = VBox(
            [tile_image],
            layout=Layout(
                width=f"{img_size}px",
                height=f"{img_size}px",
                overflow="hidden",
            ),
        )

        btn_size = "28px"
        btn_height = "24px"

        rank_label = HTML(
            value=(
                f'<span style="color:#64748b;font-size:10px;font-weight:600;">'
                f"#{rank}</span>"
                if rank
                else ""
            ),
            layout=Layout(width="24px", margin="0"),
        )

        map_button = Button(
            icon="fa-crosshairs",
            layout=Layout(width=btn_size, height=btn_height, margin="0", padding="0"),
            tooltip="Center map on tile",
        )
        map_button.add_class("tile-action-btn")
        map_button.on_click(lambda _b, r=row: self.on_center(r))

        tick_button = Button(
            icon="fa-thumbs-up",
            layout=Layout(width=btn_size, height=btn_height, margin="0", padding="0"),
            tooltip="Mark as similar",
        )
        tick_button.add_class("tile-action-btn")

        cross_button = Button(
            icon="fa-thumbs-down",
            layout=Layout(width=btn_size, height=btn_height, margin="0", padding="0"),
            tooltip="Mark as different",
        )
        cross_button.add_class("tile-action-btn")

        self._apply_label_style(point_id, tick_button, cross_button)

        tick_button.on_click(
            lambda _b, pid=point_id, r=row: self._handle_label_click(
                pid, r, UIConstants.POSITIVE_LABEL, tick_button, cross_button
            )
        )
        cross_button.on_click(
            lambda _b, pid=point_id, r=row: self._handle_label_click(
                pid, r, UIConstants.NEGATIVE_LABEL, tick_button, cross_button
            )
        )

        button_row = HBox(
            [rank_label, map_button, tick_button, cross_button],
            layout=Layout(
                justify_content="space-between",
                align_items="center",
                width="100%",
                height="28px",
                padding="2px 6px",
                background="#f8fafc",
                border_radius="0 0 6px 6px",
            ),
        )

        tile_widget = VBox(
            [image_container, button_row],
            layout=Layout(
                width=f"{img_size + 4}px",
                height="148px",
                overflow="hidden",
            ),
        )
        tile_widget.add_class("tile-card")

        if point_id in self.state.pos_ids:
            tile_widget.add_class("tile-positive")
        elif point_id in self.state.neg_ids:
            tile_widget.add_class("tile-negative")

        return tile_widget

    def _make_placeholder_tile(self) -> VBox:
        img_size = 116
        image_placeholder = Label(
            value="Loading...",
            layout=Layout(
                width=f"{img_size}px",
                height=f"{img_size}px",
                background="linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%)",
                display="flex",
                align_items="center",
                justify_content="center",
                overflow="hidden",
                border_radius="6px 6px 0 0",
                color="#94a3b8",
                font_size="11px",
            ),
        )
        button_placeholder = HBox(
            layout=Layout(
                height="28px",
                width="100%",
                background="#f8fafc",
                border_radius="0 0 6px 6px",
            )
        )
        placeholder = VBox(
            [image_placeholder, button_placeholder],
            layout=Layout(
                width=f"{img_size + 4}px",
                height="148px",
                overflow="hidden",
            ),
        )
        placeholder.add_class("tile-card")
        return placeholder

    def _update_operation(self, message: Optional[str]) -> None:
        if message:
            if hasattr(self.map_manager, "set_operation"):
                self.map_manager.set_operation(message)
        else:
            if hasattr(self.map_manager, "clear_operation"):
                self.map_manager.clear_operation()

    def _update_page_info(self, loaded: int, total: int) -> None:
        self.page_info_label.value = (
            f'<span style="color:#64748b;font-size:12px;font-weight:500;">'
            f'{loaded:,} <span style="color:#94a3b8;">of</span> {total:,}'
            f"</span>"
        )

    def _handle_label_click(
        self,
        point_id: str,
        row,
        label: str,
        tick_button: Button,
        cross_button: Button,
    ) -> None:
        self.on_label(point_id, row, label)
        self._apply_label_style(point_id, tick_button, cross_button)

    def _apply_label_style(
        self,
        point_id: str,
        tick_button: Button,
        cross_button: Button,
    ) -> None:
        # Detection mode: check detection_labels
        if self.state.detection_mode:
            detection_label = self.state.detection_labels.get(point_id)
            if detection_label == 1:
                tick_button.button_style = "success"
                tick_button.layout.opacity = "1.0"
            else:
                tick_button.button_style = ""
                tick_button.layout.opacity = "0.3"

            if detection_label == 0:
                cross_button.button_style = "danger"
                cross_button.layout.opacity = "1.0"
            else:
                cross_button.button_style = ""
                cross_button.layout.opacity = "0.3"
            return

        # Normal mode: check pos_ids and neg_ids
        if point_id in self.state.pos_ids:
            tick_button.button_style = "success"
            tick_button.layout.opacity = "1.0"
        else:
            tick_button.button_style = ""
            tick_button.layout.opacity = "0.3"

        if point_id in self.state.neg_ids:
            cross_button.button_style = "danger"
            cross_button.layout.opacity = "1.0"
        else:
            cross_button.button_style = ""
            cross_button.layout.opacity = "0.3"

    def _handle_tiles_ready(self) -> None:
        if self.results_ready:
            return
        self.results_ready = True
        callback = self._tiles_ready_callback
        self._tiles_ready_callback = None
        if callback:
            callback()


__all__ = ["TilePanel"]
