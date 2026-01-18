"""State management helpers for the GeoVibes UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from geovibes.ui_config import UIConstants

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    import pandas as pd
    import geopandas as gpd


@dataclass
class AppState:
    """Mutable state shared across UI components."""

    pos_ids: List[str] = field(default_factory=list)
    neg_ids: List[str] = field(default_factory=list)
    cached_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    query_vector: Optional[np.ndarray] = None
    selection_mode: str = "point"
    current_label: str = "Positive"
    select_val: str = UIConstants.POSITIVE_LABEL
    execute_label_point: bool = True
    lasso_mode: bool = False
    polygon_drawing: bool = False
    tile_basemap: str = "MAPTILER"
    tile_page: int = 0
    tiles_per_page: int = 50
    initial_load_size: int = 20
    last_search_results_df: Optional["pd.DataFrame"] = None
    detections_with_embeddings: Optional["gpd.GeoDataFrame"] = None
    detection_mode: bool = False
    detection_data: Optional[Dict[str, Any]] = None
    detection_labels: Dict[str, int] = field(default_factory=dict)

    # Remote database loading state
    database_loading: bool = False
    database_ready: bool = True
    loading_message: str = ""

    def set_label_mode(self, label: str) -> None:
        """Update the active label mode."""
        self.current_label = label
        if label == "Positive":
            self.select_val = UIConstants.POSITIVE_LABEL
        elif label == "Negative":
            self.select_val = UIConstants.NEGATIVE_LABEL
        else:
            self.select_val = UIConstants.ERASE_LABEL

    def reset(self) -> None:
        """Reset all mutable state."""
        self.pos_ids.clear()
        self.neg_ids.clear()
        self.cached_embeddings.clear()
        self.query_vector = None
        self.last_search_results_df = None
        self.detections_with_embeddings = None
        self.tile_page = 0
        self.tile_basemap = "MAPTILER"
        self.selection_mode = "point"
        self.set_label_mode("Positive")
        self.lasso_mode = False
        self.polygon_drawing = False
        self.detection_mode = False
        self.detection_data = None
        self.detection_labels.clear()
        self.database_loading = False
        self.database_ready = True
        self.loading_message = ""

    def toggle_label(self, point_id: str, label: str) -> None:
        """Toggle the label for a point_id, ensuring exclusivity."""
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)

        if label == UIConstants.POSITIVE_LABEL:
            self.pos_ids.append(point_id)
        elif label == UIConstants.NEGATIVE_LABEL:
            self.neg_ids.append(point_id)

    def remove_label(self, point_id: str) -> None:
        """Remove any label associated with the point."""
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)

    def update_query_vector(self) -> Optional[np.ndarray]:
        """Recompute query vector based on cached embeddings."""
        pos_embeddings = [
            self.cached_embeddings[pid]
            for pid in self.pos_ids
            if pid in self.cached_embeddings
        ]
        if not pos_embeddings:
            self.query_vector = None
            return None

        pos_vec = np.mean(pos_embeddings, axis=0)

        neg_embeddings = [
            self.cached_embeddings[nid]
            for nid in self.neg_ids
            if nid in self.cached_embeddings
        ]
        if neg_embeddings:
            neg_vec = np.mean(neg_embeddings, axis=0)
        else:
            neg_vec = np.zeros_like(pos_vec)

        self.query_vector = 2 * pos_vec - neg_vec
        return self.query_vector

    def apply_label(self, point_id: str, label: str) -> str:
        """Toggle label assignment for a point and return resulting state."""
        was_pos = point_id in self.pos_ids
        was_neg = point_id in self.neg_ids

        self.remove_label(point_id)

        if label == UIConstants.POSITIVE_LABEL and not was_pos:
            self.pos_ids.append(point_id)
            return "positive"
        if label == UIConstants.NEGATIVE_LABEL and not was_neg:
            self.neg_ids.append(point_id)
            return "negative"
        return "removed"

    def enter_detection_mode(self, detection_geojson: Dict[str, Any]) -> None:
        """Enter detection review mode with provided GeoJSON data."""
        self.detection_mode = True
        self.detection_data = detection_geojson
        self.detection_labels.clear()

    def exit_detection_mode(self) -> None:
        """Exit detection review mode and clear detection state."""
        self.detection_mode = False
        self.detection_data = None
        self.detection_labels.clear()

    def label_detection(self, tile_id: str, label: int) -> None:
        """Add or update label for a detection tile_id."""
        self.detection_labels[tile_id] = label

    def get_labeled_detections(self) -> List[Tuple[str, int]]:
        """Return list of (tile_id, label) tuples for all labeled detections."""
        return list(self.detection_labels.items())


__all__ = ["AppState"]
