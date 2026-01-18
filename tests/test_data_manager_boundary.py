import faiss

from geovibes.ui.data_manager import DataManager


def test_switch_database_updates_boundary_path(monkeypatch):
    dm = DataManager.__new__(DataManager)
    dm.verbose = False
    dm.ee_available = True
    dm.current_database_path = "db1"
    dm.current_database_info = {"faiss_path": "index1", "geometry_path": "geom1"}
    dm.current_faiss_path = "index1"
    dm.current_geometry_path = "geom1"
    dm.database_info_by_path = {
        "db2": {"faiss_path": "index2", "geometry_path": "geom2"}
    }
    dm._owns_connection = False
    dm.duckdb_connection = None

    monkeypatch.setattr(DataManager, "_connect_duckdb", lambda self, path: "conn")
    monkeypatch.setattr(DataManager, "_apply_duckdb_settings", lambda self, path: None)
    monkeypatch.setattr(DataManager, "_detect_embedding_dim", lambda self: 1)
    monkeypatch.setattr(DataManager, "_warm_up_remote_database", lambda self: None)
    monkeypatch.setattr(faiss, "read_index", lambda path: f"loaded:{path}")

    def fake_setup_boundary_and_center(self):
        return "geom2", (0.0, 0.0)

    monkeypatch.setattr(
        DataManager, "_setup_boundary_and_center", fake_setup_boundary_and_center
    )

    dm.switch_database("db2")

    assert dm.current_database_path == "db2"
    assert dm.current_geometry_path == "geom2"
    assert dm.effective_boundary_path == "geom2"
