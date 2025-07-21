from pydantic import BaseModel
from pydantic_settings import BaseSettings
import yaml
from datetime import datetime
from pathlib import Path


class GeovibesEnv(BaseSettings):
    maptiler_api_key: str = ""
    gcs_project: str = ""
    gcs_access_key_id: str = ""
    gcs_secret_access_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class AOI(BaseModel):
    name: str
    boundary: str
    dbs: dict[str, str]  # db name -> db path
    basemap_tile_server_url: str | None = None


class Options(BaseModel):
    ee_basemap_start_date: datetime = datetime(2024, 1, 1)
    ee_basemap_end_date: datetime = datetime(2025, 1, 1)
    verbose: bool = False
    local_download_path: Path = Path("local_downloads")


class Config(BaseModel):
    aois: list[AOI]
    options: Options


def load_config_from_yaml(
    config_path: str | Path,
) -> Config:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "embeddings" not in config:
        raise ValueError("No embeddings found in config")
    if "options" not in config:
        raise ValueError("No options found in config")

    aois = config["embeddings"]
    aois = [AOI(**aoi) for aoi in aois]

    options = config["options"]
    options = Options(**options)

    return Config(aois=aois, options=options)
