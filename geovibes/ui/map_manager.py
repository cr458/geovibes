"""Map construction and layer management for GeoVibes."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import geopandas as gpd
import ipyleaflet as ipyl
import ipywidgets as ipyw
import shapely.geometry
import shapely.geometry.base
from ipyleaflet import DrawControl, Map
import ipyvuetify as v
from ipywidgets import HTML, HBox, Layout, VBox
from geovibes.ee_tools import get_ee_image_url
from geovibes.ui_config import BasemapConfig, LayerStyles, UIConstants

from .status import StatusBus


class MapManager:
    """Responsible for the ipyleaflet map, layers, and status widgets."""

    def __init__(
        self,
        *,
        data_manager,
        state,
        status_bus: StatusBus,
        verbose: bool = False,
    ) -> None:
        self.data = data_manager
        self.state = state
        self.status_bus = status_bus
        self.verbose = verbose

        self.basemap_tiles = self._setup_basemap_tiles()
        self.current_basemap = "MAPTILER"
        basemap_url = self.basemap_tiles[self.current_basemap]
        self.basemap_layer = ipyl.TileLayer(
            url=basemap_url,
            no_wrap=True,
            name="basemap",
            attribution="",
        )

        self.map = self._build_map(center=(self.data.center_y, self.data.center_x))
        self.legend = self._build_legend()
        self.status_bar = HTML(value="Ready")
        self.vector_layer = None
        self.highlight_layer = None
        self._overlay_layers: Dict[str, ipyl.TileLayer] = {}
        self._geojson_layers: Dict[str, dict] = {}

        self._add_map_layers()
        self.draw_control = self._setup_draw_control()
        self._layer_manager_control = self._build_layer_manager()
        self.update_status()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _setup_basemap_tiles(self) -> Dict[str, str]:
        return BasemapConfig.BASEMAP_TILES.copy()

    def _build_map(self, center) -> Map:
        map_widget = Map(
            basemap=self.basemap_layer,
            center=center,
            zoom=UIConstants.DEFAULT_ZOOM,
            layout=Layout(flex="1 1 auto", height="100%"),
            scroll_wheel_zoom=True,
            attribution_control=False,
        )
        attribution_control = ipyl.AttributionControl(
            position="bottomleft",
            prefix='<a href="https://leafletjs.com">Leaflet</a> | '
            + BasemapConfig.MAPTILER_ATTRIBUTION,
        )
        map_widget.add_control(attribution_control)
        return map_widget

    @staticmethod
    def _build_legend() -> HTML:
        similarity_html = UIConstants.similarity_legend_html()
        legend_html = (
            "<div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>"
            "<div><strong>Labels:</strong> "
            f"<span style='color: {UIConstants.NEG_COLOR}; font-weight: bold;'>🟠 Negative</span> | "
            f"<span style='color: {UIConstants.POS_COLOR}; font-weight: bold;'>🔵 Positive</span>"
            "</div>"
            f"{similarity_html}"
            "</div>"
        )
        return HTML(value=legend_html)

    def _add_map_layers(self) -> None:
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.POS_COLOR),
        )
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.NEG_COLOR),
        )
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_erase_style(),
        )
        self.points_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_search_style(),
            hover_style=LayerStyles.get_search_hover_style(),
        )
        self.detection_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
        )

        for layer in [
            self.pos_layer,
            self.neg_layer,
            self.erase_layer,
            self.points_layer,
            self.detection_layer,
        ]:
            self.map.add_layer(layer)

    def _setup_draw_control(self) -> DrawControl:
        draw_control = DrawControl(
            polygon=LayerStyles.get_draw_options(),
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        draw_control.clear()
        self.map.add_control(draw_control)
        return draw_control

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_draw_handler(self, handler: Callable) -> None:
        self.draw_control.on_draw(handler)

    def make_layout(self, side_panel: ipyw.Widget) -> ipyw.Widget:
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
        return HBox(
            [side_panel, map_with_overlays],
            layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width="100%"),
        )

    def update_status(
        self, lat: Optional[float] = None, lon: Optional[float] = None
    ) -> None:
        if lat is None or lon is None:
            center = self.map.center
            lat, lon = center[0], center[1]
        if self.state.geo_filter_mode:
            mode = "Filter"
        elif self.state.lasso_mode:
            mode = "Polygon"
        else:
            mode = "Point"
        self.status_bar.value = self.status_bus.render(
            lat=lat,
            lon=lon,
            mode=mode,
            label=self.state.current_label if not self.state.geo_filter_mode else "N/A",
            polygon_drawing=self.state.polygon_drawing,
        )

    def set_operation(self, message: str) -> None:
        self.status_bus.set_operation(message)
        self.update_status()

    def clear_operation(self) -> None:
        self.status_bus.clear_operation()
        self.update_status()

    def update_basemap(self, basemap_name: str) -> None:
        if basemap_name not in self.basemap_tiles:
            raise ValueError(f"Unknown basemap: {basemap_name}")
        self.current_basemap = basemap_name
        self.basemap_layer.url = self.basemap_tiles[basemap_name]

    def add_widget_control(
        self, widget: ipyw.Widget, position: str = "topright"
    ) -> ipyl.WidgetControl:
        control = ipyl.WidgetControl(widget=widget, position=position)
        self.map.add_control(control)
        return control

    def update_boundary_layer(self, path: Optional[str]) -> None:
        for layer in list(self.map.layers):
            if getattr(layer, "name", None) == "region":
                self.map.remove_layer(layer)
        if not path:
            return
        try:
            boundary_gdf = gpd.read_file(path)
            boundary_geojson = json.loads(boundary_gdf.to_json())
            region_layer = ipyl.GeoJSON(
                name="region",
                data=boundary_geojson,
                style=LayerStyles.get_region_style(),
            )
            self.map.add_layer(region_layer)
            if self.verbose:
                print("✅ Boundary layer added successfully")
        except Exception as exc:
            if self.verbose:
                print(f"❌ Could not add boundary layer: {exc}")

    def set_vector_layer(
        self, geojson_data: dict, name: str, style: Optional[dict] = None
    ) -> None:
        if self.vector_layer and self.vector_layer in self.map.layers:
            old_name = self.vector_layer.name
            self.map.remove_layer(self.vector_layer)
            self._geojson_layers.pop(old_name, None)
        style = style or {
            "color": "#FF6B6B",
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#FF6B6B",
            "fillOpacity": 0.3,
        }
        self.vector_layer = ipyl.GeoJSON(name=name, data=geojson_data, style=style)
        self.map.add_layer(self.vector_layer)
        self._geojson_layers[name] = {
            "layer": self.vector_layer,
            "base_style": style.copy(),
            "opacity": 1.0,
        }
        self._refresh_layer_manager()

    def clear_vector_layer(self) -> None:
        if self.vector_layer and self.vector_layer in self.map.layers:
            name = self.vector_layer.name
            self.map.remove_layer(self.vector_layer)
            self._geojson_layers.pop(name, None)
            self._refresh_layer_manager()
        self.vector_layer = None

    def highlight_polygon(
        self,
        polygon: shapely.geometry.base.BaseGeometry,
        *,
        color: str = "yellow",
        fill_opacity: float = 0.0,
    ):
        if self.highlight_layer and self.highlight_layer in self.map.layers:
            self.map.remove_layer(self.highlight_layer)
        self.highlight_layer = ipyl.GeoJSON(
            data={
                "type": "Feature",
                "geometry": shapely.geometry.mapping(polygon),
            },
            name="tile_highlight",
            style={"color": color, "fillOpacity": fill_opacity, "weight": 3},
        )
        self.map.add_layer(self.highlight_layer)

    def clear_highlight(self) -> None:
        if self.highlight_layer and self.highlight_layer in self.map.layers:
            self.map.remove_layer(self.highlight_layer)
        self.highlight_layer = None

    def update_label_layers(
        self,
        *,
        pos_geojson: dict,
        neg_geojson: dict,
        erase_geojson: dict,
    ) -> None:
        self.pos_layer.data = pos_geojson
        self.neg_layer.data = neg_geojson
        self.erase_layer.data = erase_geojson

    def update_search_layer(
        self,
        geojson_data: dict,
        style_callback: Optional[Callable] = None,
    ) -> None:
        self.points_layer.data = geojson_data
        if style_callback:
            self.points_layer.style_callback = style_callback

    def center_on(self, lat: float, lon: float, zoom: Optional[int] = None) -> None:
        self.map.center = (lat, lon)
        if zoom is not None:
            self.map.zoom = zoom

    def update_detection_layer(
        self,
        geojson_data: dict,
        style_callback: Optional[Callable] = None,
    ) -> None:
        # Clear base style to let style_callback take full control
        self.detection_layer.style = {}

        if style_callback:
            self.detection_layer.style_callback = style_callback
        else:

            def default_style(feature):
                probability = feature.get("properties", {}).get("probability", 0.5)
                color = LayerStyles.probability_to_color(probability)
                return {
                    "color": color,
                    "weight": 2,
                    "opacity": 0.8,
                    "fillColor": color,
                    "fillOpacity": 0.1,
                }

            self.detection_layer.style_callback = default_style

        # Set data after style to trigger re-render
        self.detection_layer.data = geojson_data

    def clear_detection_layer(self) -> None:
        self.detection_layer.data = json.loads(
            gpd.GeoDataFrame(columns=["geometry"]).to_json()
        )

    # ------------------------------------------------------------------
    # Layer manager widget
    # ------------------------------------------------------------------

    def _build_layer_manager(self) -> ipyl.WidgetControl:
        self._layer_rows = v.Col(children=[], class_="pa-0 ma-0")
        self._layer_manager_container = v.Card(
            children=[
                v.CardTitle(children=["Layers"], class_="pa-2 text-subtitle-2"),
                self._layer_rows,
            ],
            class_="ma-0",
            style_="min-width: 160px; max-width: 160px;",
            elevation=2,
        )
        control = ipyl.WidgetControl(
            widget=self._layer_manager_container,
            position="topleft",
        )
        self._layer_manager_container.hide()
        self.map.add_control(control)
        return control

    def _refresh_layer_manager(self) -> None:
        rows = []
        for name in self._overlay_layers:
            layer = self._overlay_layers[name]
            row = self._create_layer_row(name, layer.opacity, layer_type="tile")
            rows.append(row)
        for name, info in self._geojson_layers.items():
            row = self._create_layer_row(name, info["opacity"], layer_type="geojson")
            rows.append(row)
        self._layer_rows.children = rows
        if rows:
            self._layer_manager_container.show()
        else:
            self._layer_manager_container.hide()

    def _create_layer_row(
        self, name: str, opacity: float, layer_type: str = "tile"
    ) -> v.Row:
        display_name = name[:9] if len(name) > 9 else name

        slider = v.Slider(
            v_model=opacity,
            min=0,
            max=1,
            step=0.05,
            hide_details=True,
            dense=True,
            class_="ma-0 pa-0",
            style_="flex: 1; min-width: 40px;",
        )

        def on_opacity_change(widget, event, data, layer_name=name, ltype=layer_type):
            if ltype == "tile" and layer_name in self._overlay_layers:
                self._overlay_layers[layer_name].opacity = data
            elif ltype == "geojson" and layer_name in self._geojson_layers:
                self._set_geojson_opacity(layer_name, data)

        slider.on_event("input", on_opacity_change)

        remove_btn = v.Btn(
            icon=True,
            x_small=True,
            children=[v.Icon(children=["mdi-close"], x_small=True)],
            color="error",
            class_="ma-0 pa-0",
            style_="min-width: 20px; width: 20px;",
        )

        def on_remove(widget, event, data, layer_name=name, ltype=layer_type):
            if ltype == "tile":
                self.remove_layer(layer_name)
            elif ltype == "geojson":
                self._remove_geojson_layer(layer_name)

        remove_btn.on_event("click", on_remove)

        return v.Row(
            children=[
                v.Html(
                    tag="span",
                    children=[display_name],
                    class_="text-caption",
                    style_=(
                        "width: 54px; min-width: 54px; flex-shrink: 0; "
                        "overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    ),
                ),
                slider,
                remove_btn,
            ],
            class_="ma-0 pa-1 align-center",
            dense=True,
            no_gutters=True,
        )

    def _set_geojson_opacity(self, name: str, opacity: float) -> None:
        if name not in self._geojson_layers:
            return
        info = self._geojson_layers[name]
        info["opacity"] = opacity
        base_style = info["base_style"]
        layer = info["layer"]
        new_style = base_style.copy()
        if "opacity" in new_style:
            new_style["opacity"] = base_style["opacity"] * opacity
        if "fillOpacity" in new_style:
            new_style["fillOpacity"] = base_style["fillOpacity"] * opacity
        layer.style = new_style

    def _remove_geojson_layer(self, name: str) -> None:
        if name not in self._geojson_layers:
            return
        info = self._geojson_layers.pop(name)
        layer = info["layer"]
        if layer in self.map.layers:
            self.map.remove_layer(layer)
        if self.vector_layer == layer:
            self.vector_layer = None
        self._refresh_layer_manager()

    # ------------------------------------------------------------------
    # Overlay tile layer management
    # ------------------------------------------------------------------

    def add_tile_layer(
        self,
        url: str,
        name: str,
        opacity: float = 1.0,
        attribution: str = "",
    ) -> None:
        if name in self._overlay_layers:
            self.remove_layer(name)
        opacity = max(0.0, min(1.0, opacity))
        layer = ipyl.TileLayer(
            url=url,
            name=name,
            opacity=opacity,
            attribution=attribution,
            no_wrap=True,
        )
        self._overlay_layers[name] = layer
        self._insert_overlay_layer(layer)
        self._refresh_layer_manager()

    def add_ee_layer(
        self,
        ee_image,
        vis_params: Dict,
        name: str,
        opacity: float = 1.0,
    ) -> None:
        if not self.data.ee_available:
            raise RuntimeError("Earth Engine not available")
        url = get_ee_image_url(ee_image, vis_params)
        self.add_tile_layer(url, name, opacity, attribution="Google Earth Engine")

    def remove_layer(self, name: str) -> bool:
        if name not in self._overlay_layers:
            return False
        layer = self._overlay_layers.pop(name)
        if layer in self.map.layers:
            self.map.remove_layer(layer)
        self._refresh_layer_manager()
        return True

    def set_layer_opacity(self, name: str, opacity: float) -> None:
        if name not in self._overlay_layers:
            raise ValueError(f"Layer '{name}' not found")
        self._overlay_layers[name].opacity = max(0.0, min(1.0, opacity))

    def list_overlay_layers(self) -> List[str]:
        return list(self._overlay_layers.keys())

    def clear_overlay_layers(self) -> None:
        for name in list(self._overlay_layers.keys()):
            layer = self._overlay_layers.pop(name)
            if layer in self.map.layers:
                self.map.remove_layer(layer)
        self._refresh_layer_manager()

    def _insert_overlay_layer(self, layer: ipyl.TileLayer) -> None:
        layers = list(self.map.layers)
        basemap_idx = 0
        for i, lyr in enumerate(layers):
            if lyr is self.basemap_layer:
                basemap_idx = i
                break
        # Insert after basemap and all existing overlays (new layers appear on top)
        # Note: layer is already in _overlay_layers when this is called
        insert_idx = basemap_idx + len(self._overlay_layers)
        layers.insert(insert_idx, layer)
        self.map.layers = tuple(layers)


__all__ = ["MapManager"]
