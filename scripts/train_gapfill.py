"""Train the gap-filling models served by the API (part 3), on every plot of the chosen seasons.

One XGBoost model per crop, per mode (after the season / real time) and with or without radar:
8 models, with the inputs chosen in notebook 5 (`SERVED_FEATURES`), saved as JSON in
models/gapfill/ with a metadata file. 15 % of the plots of each season are held out to stop the
training and to report a validation error. That error comes from the training seasons, so it
favours models that use the date; notebook 5 (section 13) compares the models on a new season.

Usage: python scripts/train_gapfill.py --seasons 2023-24   (after scripts/extract_part2.py)

The deployed models are trained on 2023-24 only, so that the web app's demo season, 2024-25, is one
they have never seen: the real-life setup of notebook 5. In production, train on every past season.
"""

import argparse
import json
import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sentinel_agri.config import MODELS_DIR, SEASONS
from sentinel_agri.gapfill import (
    SERVED_FEATURES,
    build_grid,
    despike,
    make_masks,
    tabular,
)
from sentinel_agri.parcels import CROPS, select_parcels
from sentinel_agri.timeseries import s1_backscatter, s2_ndvi

OUT_DIR = MODELS_DIR / "gapfill"
QUANTILES = (0.1, 0.9)
TARGET_COVERAGE = 0.8


def slug(crop: str) -> str:
    return crop.lower().replace(" ", "_")


def season_tables(season: str, seed: int = 0):
    parcels = select_parcels(season)
    grid = build_grid(s2_ndvi(parcels, season), s1_backscatter(parcels, season), parcels, season)
    grid.ndvi, _ = despike(grid.ndvi)
    rows = np.random.default_rng(seed).permutation(len(grid.parcel_ids))
    stop, fit = rows[: len(rows) * 15 // 100], rows[len(rows) * 15 // 100 :]

    def gaps(r, s):
        hidden = np.zeros_like(grid.observed)
        hidden[r] = make_masks(grid.observed[r], seed=s)
        return hidden

    fit_table = pd.concat(
        [tabular(grid, fit, gaps(fit, 300 + k)) for k in range(5)], ignore_index=True
    )
    # The held-out plots get two different sets of gaps: one stops the training, the other
    # calibrates the uncertainty ranges on values no model has used.
    stop_table, calibration_table = (
        tabular(grid, stop, gaps(stop, 400)),
        tabular(grid, stop, gaps(stop, 401)),
    )
    return fit_table, stop_table, calibration_table, len(rows)


def train_interval(fit, stop, calibration, features, path) -> tuple[float, float]:
    """A model of the 10th and 90th percentiles of NDVI, then conformal calibration.

    Quantile models learn how uncertain they are on their training data, and are overconfident
    elsewhere. Conformal calibration measures, on held-out values, by how much the range must widen
    to contain the truth TARGET_COVERAGE of the time, and the API adds that margin on both sides.
    On the unseen 2024-25 season, this brings the coverage from 62-72 % to 75-79 %.
    """
    model = XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES),
        early_stopping_rounds=100, random_state=0,
    )  # fmt: skip
    model.fit(
        fit[features], fit["target"], eval_set=[(stop[features], stop["target"])], verbose=False
    )
    model.save_model(path)
    low, high = model.predict(calibration[features]).T
    scores = np.maximum(low - calibration["target"], calibration["target"] - high)
    widen = float(np.quantile(scores, min(1.0, TARGET_COVERAGE * (1 + 1 / len(scores)))))
    return widen, TARGET_COVERAGE


def main() -> None:
    warnings.filterwarnings("ignore", message=".*no geotransform.*")
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seasons", nargs="+", choices=list(SEASONS), default=list(SEASONS))
    seasons = parser.parse_args().seasons
    tables = [season_tables(season) for season in seasons]
    fit = pd.concat([t[0] for t in tables], ignore_index=True)
    stop = pd.concat([t[1] for t in tables], ignore_index=True)
    calibration = pd.concat([t[2] for t in tables], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "trained_on": seasons,
        "plots": int(sum(t[3] for t in tables)),
        "interval": f"{QUANTILES[0]:.0%}-{QUANTILES[1]:.0%} quantiles, conformally calibrated",
        "date": date.today().isoformat(),
        "models": {},
    }
    for crop in CROPS.values():
        f, s = fit[fit["crop"] == crop], stop[stop["crop"] == crop]
        c = calibration[calibration["crop"] == crop]
        for (mode, radar), features in SERVED_FEATURES.items():
            features = [x for x in features if x != "rapeseed"]  # constant within one crop
            model = XGBRegressor(
                n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, objective="reg:absoluteerror", early_stopping_rounds=100,
                random_state=0,
            )  # fmt: skip
            model.fit(
                f[features], f["target"], eval_set=[(s[features], s["target"])], verbose=False
            )
            name = f"{mode}_{slug(crop)}_{'radar' if radar else 'no_radar'}"
            model.save_model(OUT_DIR / f"{name}.json")
            mae = float(np.abs(model.predict(s[features]) - s["target"]).mean())
            widen, coverage = train_interval(f, s, c, features, OUT_DIR / f"{name}_interval.json")
            metadata["models"][name] = {
                "crop": crop,
                "mode": mode,
                "radar": radar,
                "features": features,
                "validation_mae": round(mae, 4),
                "interval_model": f"{name}_interval.json",
                "interval_widen": round(widen, 4),
                "interval_coverage_target": TARGET_COVERAGE,
            }
            print(f"{name:<45} validation MAE {mae:.4f}, range widened by {widen:.3f}")
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(metadata['models'])} models to {OUT_DIR}")


if __name__ == "__main__":
    main()
