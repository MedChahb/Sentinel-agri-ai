"""Sentinel Agri AI API (part 3).

- `POST /classify`: land-cover class of a 64 x 64 Sentinel-2 Level-1C tile (part 1 model, ONNX).
- `POST /gapfill`: a farm plot's NDVI series with its cloud gaps filled (part 2 models, XGBoost).
- `GET /health`: service status and which models are loaded.

Models are read at start-up; a missing model only disables its endpoint (503).
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import onnxruntime as ort
import pandas as pd
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from rasterio.io import MemoryFile

from sentinel_agri.gapfill import build_grid, load_served, tabular

CLASSIFIER_PATH = Path(os.environ.get("MODEL_PATH", "models/eurosat_resnet18.onnx"))
GAPFILL_DIR = Path(os.environ.get("GAPFILL_DIR", "models/gapfill"))
TILE_SIZE = 64
# Band order of the uploaded tiles (EuroSAT files).
INPUT_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
]
N_BANDS = len(INPUT_BANDS)
MIN_RADAR_PASSES = 10  # below this, the gap-filling models without radar are used

state: dict = {}


def load_classifier(path: Path) -> dict | None:
    if not path.exists():
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    metadata = json.loads(path.with_suffix(".json").read_text())
    return {"session": session, "classes": metadata["classes"]}


def load_gapfill(directory: Path) -> dict | None:
    if not (directory / "metadata.json").exists():
        return None
    metadata = json.loads((directory / "metadata.json").read_text())
    return {"models": load_served(directory), "trained_on": metadata["trained_on"]}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["classifier"] = load_classifier(CLASSIFIER_PATH)
    state["gapfill"] = load_gapfill(GAPFILL_DIR)
    yield
    state.clear()


app = FastAPI(title="Sentinel Agri AI", version="0.2.0", lifespan=lifespan)


class Health(BaseModel):
    status: str
    model_loaded: bool  # land-cover classifier
    gapfill_loaded: bool


@app.get("/health")
def health() -> Health:
    return Health(
        status="ok",
        model_loaded=state.get("classifier") is not None,
        gapfill_loaded=state.get("gapfill") is not None,
    )


# --- Land-cover classification ------------------------------------------------------------------


class Prediction(BaseModel):
    label: str
    probability: float
    probabilities: dict[str, float]
    warning: str | None = None  # set when a Level-2A tile was classified on request


def read_tile(data: bytes) -> np.ndarray:
    try:
        with MemoryFile(data) as memfile, memfile.open() as src:
            tile = src.read().astype(np.float32)
    except rasterio.errors.RasterioIOError as err:
        raise HTTPException(422, "The file is not a readable GeoTIFF.") from err
    if tile.shape != (N_BANDS, TILE_SIZE, TILE_SIZE):
        expected = (N_BANDS, TILE_SIZE, TILE_SIZE)
        raise HTTPException(422, f"Expected a tile of shape {expected}, got {tile.shape}.")
    return tile


# The model was trained on Level-1C and fails on Level-2A (notebook 3: 44.6 % instead of 96.6 %).
# Both checks below flag none of the 28,697 Level-1C tiles tested (EuroSAT and 2025 patches).
MIN_B10_STD = 0.1  # Level-2A has no B10, so a sender has to fill it, usually with a constant
MIN_B01_B02_RATIO = 0.85  # haze keeps B01 above 0.93 x B02 in Level-1C; 69 % of Level-2A is below


LEVEL_2A_REFUSED = (
    "The tile looks like Level-2A (atmospherically corrected) data. This model only supports "
    "Level-1C, top-of-atmosphere reflectance. Add ?allow_level2a=true to classify it anyway."
)
LEVEL_2A_WARNING = (
    "This tile looks like Level-2A, but the model was trained on Level-1C only, which is not "
    "enough to generalise to Level-2A: on real Level-2A tiles it was right 44.6 % of the time "
    "(96.6 % on Level-1C), often with high confidence. Treat this answer as unreliable."
)


def looks_like_level_2a(tile: np.ndarray) -> bool:
    b01, b02, b10 = (tile[INPUT_BANDS.index(b)] for b in ("B01", "B02", "B10"))
    ratio = float(np.median(b01)) / max(float(np.median(b02)), 1.0)
    return bool(b10.std() < MIN_B10_STD or ratio < MIN_B01_B02_RATIO)


@app.post("/classify")
async def classify(file: Annotated[UploadFile, File()], allow_level2a: bool = False) -> Prediction:
    """Land-cover class of a Level-1C tile.

    Level-2A tiles are refused by default, so that a program does not silently receive confident
    wrong answers. With `allow_level2a=true`, they are classified and the answer carries a warning.
    """
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(
            503, "No classifier loaded. Export it with scripts/export_classifier.py."
        )
    tile = read_tile(await file.read())
    level_2a = looks_like_level_2a(tile)
    if level_2a and not allow_level2a:
        raise HTTPException(422, LEVEL_2A_REFUSED)
    logits = classifier["session"].run(None, {"tiles": tile[None]})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    best = int(probs.argmax())
    return Prediction(
        label=classifier["classes"][best],
        probability=float(probs[best]),
        probabilities={c: float(p) for c, p in zip(classifier["classes"], probs, strict=True)},
        warning=LEVEL_2A_WARNING if level_2a else None,
    )


# --- NDVI gap-filling ---------------------------------------------------------------------------


class NdviObservation(BaseModel):
    date: date
    ndvi: float = Field(ge=-1, le=1, description="Mean NDVI of the plot on a clear date")


class RadarObservation(BaseModel):
    date: date
    vv_db: float = Field(description="Mean VV backscatter of the plot, in dB")
    vh_db: float = Field(description="Mean VH backscatter of the plot, in dB")
    orbit: int = Field(
        0, description="Sentinel-1 relative orbit; offsets between orbits are removed"
    )


class GapfillRequest(BaseModel):
    crop: Literal["Winter wheat", "Winter rapeseed"]
    mode: Literal["after_season", "real_time"] = Field(
        "after_season",
        description="after_season: clear values on both sides of a gap; real_time: only the past",
    )
    ndvi: list[NdviObservation] = Field(min_length=2, description="Clear observations only")
    radar: list[RadarObservation] = Field(default_factory=list, description="Optional Sentinel-1")
    season_start: date | None = Field(None, description="Default: 1 October before the first value")


class GapfillResponse(BaseModel):
    dates: list[date]
    ndvi: list[float | None]
    ndvi_low: list[float | None] = Field(
        description="Calibrated range: the truth falls between low and high about 80 % of the time"
    )
    ndvi_high: list[float | None]
    observed: list[bool]
    radar_used: bool
    mode: str


def season_bounds(request: GapfillRequest) -> tuple[date, date]:
    first = min(o.date for o in request.ndvi)
    start = request.season_start or date(first.year if first.month >= 10 else first.year - 1, 10, 1)
    return start, date(start.year + 1, 8, 31)


@app.post("/gapfill")
def gapfill(request: GapfillRequest) -> GapfillResponse:
    """Fill the gaps of one plot's NDVI series, on a 5-day grid from 1 October to 31 August."""
    service = state.get("gapfill")
    if service is None:
        raise HTTPException(
            503, "No gap-filling models loaded. Train them with scripts/train_gapfill.py."
        )
    start, end = season_bounds(request)
    if any(not start <= o.date <= end for o in request.ndvi + request.radar):
        raise HTTPException(422, f"All dates must fall within one season, {start} to {end}.")

    ndvi = pd.DataFrame({"parcel_id": 1, "date": pd.to_datetime([o.date for o in request.ndvi]),
                         "ndvi": [o.ndvi for o in request.ndvi], "clear_share": 1.0})  # fmt: skip
    radar = pd.DataFrame(
        {"parcel_id": 1, "date": pd.to_datetime([o.date for o in request.radar]),
         "orbit": [o.orbit for o in request.radar], "vv_db": [o.vv_db for o in request.radar],
         "vh_db": [o.vh_db for o in request.radar]},
        columns=["parcel_id", "date", "orbit", "vv_db", "vh_db"],
    )  # fmt: skip
    radar["ratio_db"] = radar["vh_db"] - radar["vv_db"]
    parcels = pd.DataFrame({"parcel_id": [1], "crop": [request.crop]})
    grid = build_grid(ndvi, radar, parcels, start=str(start), end=str(end))

    use_radar = len(request.radar) >= MIN_RADAR_PASSES
    model = service["models"][(request.crop, request.mode, use_radar)]
    missing = ~grid.observed
    table = tabular(grid, np.array([0]), missing, require_next=request.mode == "after_season")
    filled = np.where(grid.observed[0], grid.ndvi[0], np.nan)
    low, high = filled.copy(), filled.copy()  # an observed value has no range
    if len(table):
        steps = table["step"].to_numpy()
        filled[steps], low[steps], high[steps] = model.predict(table)

    def as_list(values):
        return [None if np.isnan(v) else round(float(v), 4) for v in values]

    return GapfillResponse(
        dates=[d.date() for d in grid.dates],
        ndvi=as_list(filled),
        ndvi_low=as_list(low),
        ndvi_high=as_list(high),
        observed=grid.observed[0].tolist(),
        radar_used=use_radar,
        mode=request.mode,
    )
