"""Precompute what the web app shows (app/data/ and app/samples/), as a real pipeline would.

A monitoring service does not run its models while a user waits: after each satellite pass, a batch
job updates every registered field, and the app reads the results. This script is that batch job
for the demo area east of Reims:

- fields: the 2024-25 plots, their clear NDVI and radar, the series filled after the season and
  the real-time estimate at every 5-day step, from the models the API serves (models/gapfill/);
- land cover: the Reims area in the four seasons of 2025, classified patch by patch with the
  exported classifier, next to ESA WorldCover;
- samples: tiles for the classification page, including a Level-2A tile the API must refuse.

Usage: python scripts/build_app_data.py [--fields-only]   (after export_classifier.py and
train_gapfill.py)
Needs the cached part 2 series, the part 1 real-world cache, and the Copernicus credentials in .env
for the full Level-1C scenes (downloaded once, then cached).
"""

import json
import logging
import shutil
import sys
import warnings

import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from odc.stac import load
from PIL import Image

from sentinel_agri.config import EUROSAT_CLASSES, EUROSAT_DIR, MODELS_DIR, ROOT
from sentinel_agri.gapfill import MIN_CLEAR_SHARE, build_grid, despike, frozen_steps, tabular
from sentinel_agri.parcels import select_parcels
from sentinel_agri.realworld import (
    CACHE_DIR,
    EUROSAT_TO_GROUP,
    PATCH,
    REGIONS,
    SCL_BAD,
    WORLDCOVER,
    _catalog,
    load_l1c,
    to_geotiff,
    worldcover,
)
from sentinel_agri.timeseries import s1_backscatter, s2_ndvi

APP_DIR = ROOT / "app"
DATA_DIR = APP_DIR / "data"
SAMPLES_DIR = APP_DIR / "samples"
SEASON = "2024-25"
MIN_RADAR_PASSES = 10  # as in the API
GROUPS = ["Crop", "Grass", "Forest", "Water", "Built-up"]


# --- Fields -------------------------------------------------------------------------------------


def predict(models, grid, rows, hidden, mode, crop, radar_rows) -> pd.DataFrame:
    """Predictions at every hidden step of the given plots, with the served model of each case."""
    out = []
    for use_radar in (True, False):
        subset = rows[np.isin(rows, radar_rows) == use_radar]
        if not len(subset):
            continue
        model, features = models[(crop, mode, use_radar)]
        table = tabular(grid, subset, hidden, require_next=mode == "after_season")
        if len(table):
            values = np.clip(model.inplace_predict(table[features]), -1, 1)
            out.append(table[["parcel_id", "step"]].assign(value=values))
    return pd.concat(out, ignore_index=True)


def build_fields() -> None:
    parcels = select_parcels(SEASON)
    ndvi, s1 = s2_ndvi(parcels, SEASON), s1_backscatter(parcels, SEASON)
    grid = build_grid(ndvi, s1, parcels, SEASON)
    grid.ndvi, _ = despike(grid.ndvi)

    metadata = json.loads((MODELS_DIR / "gapfill" / "metadata.json").read_text())
    models = {
        (m["crop"], m["mode"], m["radar"]): (
            xgb.Booster(model_file=MODELS_DIR / "gapfill" / f"{name}.json"),
            m["features"],
        )
        for name, m in metadata["models"].items()
    }
    passes = s1.groupby("parcel_id").size().reindex(grid.parcel_ids, fill_value=0).to_numpy()
    radar_rows = np.flatnonzero(passes >= MIN_RADAR_PASSES)
    hidden = ~grid.observed  # every step without a clear observation

    n_plots, n_steps = grid.ndvi.shape
    series = pd.DataFrame(
        {
            "parcel_id": np.repeat(grid.parcel_ids, n_steps),
            "step": np.tile(np.arange(n_steps), n_plots).astype(np.int8),
            "observed": grid.ndvi.ravel().astype(np.float32),
        }
    )
    index = pd.MultiIndex.from_arrays([series["parcel_id"], series["step"]])
    # On frosty days, radar sees crops vanish (frozen_steps). In real time, the model without
    # radar is then far more accurate (notebook-style check on 9 January 2024: 0.04 against 0.15
    # for wheat); after the season, the clear values around the gap keep the radar model ahead.
    frozen = frozen_steps(grid)
    for mode in ("after_season", "real_time"):
        parts, fallback = [], []
        for crop in np.unique(grid.crops):
            rows = np.flatnonzero(grid.crops == crop)
            parts.append(predict(models, grid, rows, hidden, mode, crop, radar_rows))
            if mode == "real_time":
                fallback.append(predict(models, grid, rows, hidden, mode, crop, radar_rows=[]))
        predicted = pd.concat(parts).set_index(["parcel_id", "step"])["value"]
        if fallback:
            no_radar = pd.concat(fallback).set_index(["parcel_id", "step"])["value"]
            on_frost = no_radar[
                np.isin(no_radar.index.get_level_values("step"), np.flatnonzero(frozen))
            ]
            predicted.loc[on_frost.index] = on_frost
        filled = predicted.reindex(index).to_numpy()
        series[mode] = np.where(series["observed"].notna(), series["observed"], filled)
        series[mode] = series[mode].astype(np.float32)
    series.to_parquet(DATA_DIR / "series.parquet", index=False)

    # What a client sends to the API: the clear observations and the radar passes.
    clear = ndvi[ndvi["clear_share"] >= MIN_CLEAR_SHARE][["parcel_id", "date", "ndvi"]]
    clear.astype({"ndvi": np.float32}).to_parquet(DATA_DIR / "clear_ndvi.parquet", index=False)
    radar = s1[["parcel_id", "date", "orbit", "vv_db", "vh_db"]]
    radar.astype({"vv_db": np.float32, "vh_db": np.float32, "orbit": np.int16}).to_parquet(
        DATA_DIR / "radar.parquet", index=False
    )

    plots = parcels[["parcel_id", "crop", "area_ha", "geometry"]].to_crs(4326)
    plots["geometry"] = plots.geometry.simplify(0.00003)  # about 3 m: invisible on a field map
    plots["area_ha"] = plots["area_ha"].round(1)
    plots["radar_passes"] = passes
    plots.to_file(DATA_DIR / "plots.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    meta = {
        "season": SEASON,
        "dates": [d.date().isoformat() for d in grid.dates],
        "models_trained_on": metadata["trained_on"],
        "min_radar_passes": MIN_RADAR_PASSES,
        "frost_dates": [d.date().isoformat() for d in grid.dates[frozen]],
    }
    (DATA_DIR / "fields_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"fields: {n_plots} plots x {n_steps} steps, {len(radar_rows)} with radar, frost on",
        meta["frost_dates"],
    )


# --- Land cover ---------------------------------------------------------------------------------


def scl_bad_share(region, date: str) -> np.ndarray:
    """Share of cloudy or missing pixels in each patch, from the Level-2A scene classification."""
    items = (
        _catalog()
        .search(collections=["sentinel-2-l2a"], bbox=region.bbox_lonlat, datetime=f"{date}/{date}")
        .item_collection()
    )
    scl = load(items, bands=["SCL"], geobox=region.geobox, groupby="solar_day")["SCL"].values[0]
    bad = np.isin(np.nan_to_num(scl, nan=0), SCL_BAD)
    n = region.n_patches
    return bad[: n * PATCH, : n * PATCH].reshape(n, PATCH, n, PATCH).mean(axis=(1, 3))


def to_rgb_jpg(l1c: np.ndarray, path) -> None:
    """True-colour picture (B04, B03, B02 in EuroSAT file order: 3, 2, 1), half resolution.

    Each band is stretched between its own 2nd and 98th percentiles: Level-1C is top-of-atmosphere
    reflectance, and the haze it contains would otherwise give the whole picture a blue cast.
    """
    rgb = np.stack([l1c[3], l1c[2], l1c[1]], axis=-1).astype(np.float32)
    low, high = np.percentile(rgb, 2, axis=(0, 1)), np.percentile(rgb, 98, axis=(0, 1))
    rgb = np.clip((rgb - low) / (high - low), 0, 1) ** 0.9
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    image = image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS)
    image.save(path, quality=85)


def build_landcover() -> None:
    region_key, region = "marne", REGIONS["marne"]
    n = region.n_patches
    session = ort.InferenceSession(
        str(MODELS_DIR / "eurosat_resnet18.onnx"), providers=["CPUExecutionProvider"]
    )
    out_dir = DATA_DIR / "landcover"
    out_dir.mkdir(exist_ok=True)

    scenes = sorted(CACHE_DIR.glob(f"{region_key}_*.npz"))  # the dates chosen in notebook 3
    seasons = [p.stem.split("_")[1] for p in scenes]
    dates = [p.stem.split("_")[2] for p in scenes]
    order = np.argsort([["winter", "spring", "summer", "autumn"].index(s) for s in seasons])
    seasons, dates = [seasons[i] for i in order], [dates[i] for i in order]

    classes = np.zeros((len(seasons), n, n), np.int8)
    confidence = np.zeros((len(seasons), n, n), np.float16)
    cloudy = np.zeros((len(seasons), n, n), np.float16)
    for k, (season, date) in enumerate(zip(seasons, dates, strict=True)):
        cache = CACHE_DIR / f"{region_key}_full_l1c_{date}.npy"
        if not cache.exists():
            np.save(cache, load_l1c(region, date))  # about 150 MB per scene, downloaded once
        l1c = np.load(cache)
        tiles = (
            l1c[:, : n * PATCH, : n * PATCH]
            .reshape(13, n, PATCH, n, PATCH)
            .transpose(1, 3, 0, 2, 4)
            .reshape(n * n, 13, PATCH, PATCH)
            .astype(np.float32)
        )
        logits = np.concatenate(
            [session.run(None, {"tiles": tiles[i : i + 64]})[0] for i in range(0, len(tiles), 64)]
        )
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        classes[k] = probs.argmax(axis=1).reshape(n, n)
        confidence[k] = probs.max(axis=1).reshape(n, n)
        cloudy[k] = scl_bad_share(region, date)
        to_rgb_jpg(l1c[:, : n * PATCH, : n * PATCH], out_dir / f"{region_key}_{season}.jpg")
        print(f"land cover: {season} {date} classified, {(cloudy[k] > 0.3).sum()} cloudy patches")

    wc = worldcover(region)[: n * PATCH, : n * PATCH]
    group = np.full(wc.shape, -1, np.int8)
    for value, label in WORLDCOVER.items():
        group[wc == value] = GROUPS.index(label)
    blocks = group.reshape(n, PATCH, n, PATCH).transpose(0, 2, 1, 3).reshape(n, n, -1)
    counts = np.stack([(blocks == g).sum(axis=-1) for g in range(len(GROUPS))], axis=-1)
    wc_group = counts.argmax(axis=-1).astype(np.int8)
    wc_share = (counts.max(axis=-1) / blocks.shape[-1]).astype(np.float16)

    np.savez_compressed(
        out_dir / f"{region_key}.npz", classes=classes, confidence=confidence, cloudy=cloudy,
        worldcover=wc_group, worldcover_share=wc_share,
    )  # fmt: skip
    meta = {
        "region": region.name,
        "seasons": seasons,
        "dates": dates,
        "classes": EUROSAT_CLASSES,
        "class_to_group": EUROSAT_TO_GROUP,
        "groups": GROUPS,
        "bbox_lonlat": [round(v, 5) for v in region.bbox_lonlat],
        "patch_m": PATCH * 10,
    }
    (out_dir / f"{region_key}.json").write_text(json.dumps(meta, indent=1))


# --- Samples ------------------------------------------------------------------------------------


def build_samples() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    tif_dir = next(EUROSAT_DIR.rglob("tif"))
    for name in EUROSAT_CLASSES:
        shutil.copy(tif_dir / name / f"{name}_1.tif", SAMPLES_DIR / f"eurosat_{name}.tif")
    # One patch of the Reims area on 29 April 2025, as Level-1C (accepted) and Level-2A (refused).
    scene = np.load(next(CACHE_DIR.glob("marne_spring_*.npz")))
    (SAMPLES_DIR / "reims_2025-04-29_level1c.tif").write_bytes(to_geotiff(scene["l1c"][0]))
    (SAMPLES_DIR / "reims_2025-04-29_level2a.tif").write_bytes(to_geotiff(scene["l2a"][0]))
    print(f"samples: {len(list(SAMPLES_DIR.glob('*.tif')))} tiles")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*no geotransform.*")
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_fields()
    if "--fields-only" not in sys.argv:
        build_samples()
        build_landcover()


if __name__ == "__main__":
    main()
