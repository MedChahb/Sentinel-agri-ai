"""Part 2: filling cloud gaps in NDVI time series with Sentinel-1 radar.

Everything works on a common time grid of 5-day steps. For each plot:

- `ndvi[p, t]`: mean clear NDVI in step t, NaN when no clear observation exists;
- `s1[p, t, :]`: VV, VH and VH/VV in dB, orbit offsets removed, interpolated to every step.

Artificial gaps (`make_mask`) hide part of the clear observations; models predict the hidden values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sentinel_agri.config import SEASONS

STEP_DAYS = 5
MIN_CLEAR_SHARE = 0.9  # an NDVI observation counts when at least 90 % of the plot is clear
S1_VARIABLES = ["vv_db", "vh_db", "ratio_db"]


@dataclass
class Grid:
    parcel_ids: np.ndarray  # (P,)
    crops: np.ndarray  # (P,) crop names
    dates: pd.DatetimeIndex  # (T,) start of each step
    ndvi: np.ndarray  # (P, T), NaN where no clear observation
    s1: np.ndarray  # (P, T, 3) VV, VH, VH/VV in dB, interpolated

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.ndvi)

    @property
    def doy(self) -> np.ndarray:
        """Day of the year at the middle of each step."""
        return (self.dates + pd.Timedelta(days=STEP_DAYS / 2)).dayofyear.to_numpy()


def harmonise_orbits(s1: pd.DataFrame) -> pd.DataFrame:
    """Remove each orbit's offset per plot.

    The four relative orbits see the plots from different angles, so their backscatter levels
    differ. For each plot and variable, each orbit is shifted so that its mean equals the plot's.
    """
    s1 = s1.copy()
    for var in S1_VARIABLES:
        orbit_mean = s1.groupby(["parcel_id", "orbit"])[var].transform("mean")
        plot_mean = s1.groupby("parcel_id")[var].transform("mean")
        s1[var] = s1[var] - orbit_mean + plot_mean
    return s1


def build_grid(
    ndvi: pd.DataFrame,
    s1: pd.DataFrame,
    parcels: pd.DataFrame,
    season: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Grid:
    """5-day grid of one season (`season`, or explicit `start` and `end` dates, as the API uses)."""
    start, end = (pd.Timestamp(d) for d in (SEASONS[season] if season else (start, end)))
    dates = pd.date_range(start, end, freq=f"{STEP_DAYS}D")
    parcel_ids = parcels["parcel_id"].to_numpy()
    row = {pid: i for i, pid in enumerate(parcel_ids)}
    n_p, n_t = len(parcel_ids), len(dates)

    clear = ndvi[ndvi["clear_share"] >= MIN_CLEAR_SHARE]
    step = ((clear["date"] - start).dt.days // STEP_DAYS).to_numpy()
    grid_ndvi = np.full((n_p, n_t), np.nan)
    means = clear.assign(step=step).groupby(["parcel_id", "step"])["ndvi"].mean()
    for (pid, t), value in means.items():
        if pid in row and 0 <= t < n_t:
            grid_ndvi[row[pid], t] = value

    s1 = harmonise_orbits(s1)
    s1_step = ((s1["date"] - start).dt.days // STEP_DAYS).to_numpy()
    s1_means = s1.assign(step=s1_step).groupby(["parcel_id", "step"])[S1_VARIABLES].mean()
    grid_s1 = np.full((n_p, n_t, len(S1_VARIABLES)), np.nan)
    for (pid, t), values in s1_means.iterrows():
        if pid in row and 0 <= t < n_t:
            grid_s1[row[pid], t] = values.to_numpy()
    steps = np.arange(n_t)
    for p in range(n_p):  # radar every few days: fill the rare empty steps by interpolation
        for v in range(len(S1_VARIABLES)):
            known = ~np.isnan(grid_s1[p, :, v])
            if known.any():
                grid_s1[p, :, v] = np.interp(steps, steps[known], grid_s1[p, known, v])

    crops = parcels.set_index("parcel_id").loc[parcel_ids, "crop"].to_numpy()
    return Grid(parcel_ids, crops, dates, grid_ndvi, grid_s1)


# --- Cleaning -----------------------------------------------------------------------------------


def despike(ndvi: np.ndarray, threshold: float = 0.15, max_steps: int = 2):
    """Remove isolated drops that are most likely clouds or haze the cloud mask missed.

    A value is removed when it lies more than `threshold` below both neighbouring observations,
    each at most `max_steps` steps (10 days) away: one bad date between two good ones. Real changes
    such as harvest (the next value stays low) or flowering (it lasts weeks) are kept.
    Returns the cleaned array and the mask of removed values.
    """
    observed = ~np.isnan(ndvi)
    prev, nxt = neighbours(observed)
    flagged = np.zeros_like(observed)
    for p, t in zip(*np.nonzero(observed), strict=True):
        tp, tn = prev[p, t], nxt[p, t]
        if tp < 0 or tn < 0 or t - tp > max_steps or tn - t > max_steps:
            continue
        if min(ndvi[p, tp], ndvi[p, tn]) - ndvi[p, t] > threshold:
            flagged[p, t] = True
    return np.where(flagged, np.nan, ndvi), flagged


# --- Artificial gaps ---------------------------------------------------------------------------


def make_mask(
    observed: np.ndarray,
    rng: np.random.Generator,
    hidden_share: float = 0.25,
    block_steps: int = 6,
    n_blocks: int = 1,
) -> np.ndarray:
    """Hide about `hidden_share` of the observed steps of one plot.

    First `n_blocks` blocks of `block_steps` consecutive steps (6 steps = 30 days), as long cloudy
    spells do, then single steps at random. The first and last observations stay visible, so every
    hidden value lies between two visible ones.
    """
    hidden = np.zeros_like(observed)
    obs = np.flatnonzero(observed)
    if len(obs) < 6:
        return hidden
    for _ in range(n_blocks):
        if obs[-1] - obs[0] > block_steps + 1:
            start = rng.integers(obs[0] + 1, obs[-1] - block_steps)
            hidden[start : start + block_steps] = True
    hidden &= observed
    hidden[obs[0]] = hidden[obs[-1]] = False
    target = int(round(hidden_share * len(obs)))
    candidates = [t for t in obs[1:-1] if not hidden[t]]
    extra = max(0, target - int(hidden.sum()))
    if extra and candidates:
        hidden[rng.choice(candidates, size=min(extra, len(candidates)), replace=False)] = True
    return hidden


def make_masks(observed: np.ndarray, seed: int, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([make_mask(row, rng, **kwargs) for row in observed])


def neighbours(visible: np.ndarray):
    """For every step, the index of the previous and next visible step (-1 if none)."""
    n_p, n_t = visible.shape
    prev = np.full((n_p, n_t), -1)
    nxt = np.full((n_p, n_t), -1)
    for p in range(n_p):
        last = -1
        for t in range(n_t):
            prev[p, t] = last
            if visible[p, t]:
                last = t
        last = -1
        for t in range(n_t - 1, -1, -1):
            nxt[p, t] = last
            if visible[p, t]:
                last = t
    return prev, nxt


def gap_days(hidden: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Length in days of the gap around each step: from the previous to the next visible step."""
    prev, nxt = neighbours(visible)
    return np.where((prev >= 0) & (nxt >= 0), (nxt - prev) * STEP_DAYS, -1)


# --- Baselines ---------------------------------------------------------------------------------


def linear_fill(ndvi: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Straight line between the visible observations of each plot."""
    steps = np.arange(ndvi.shape[1])
    out = np.full(ndvi.shape, np.nan)
    for p in range(ndvi.shape[0]):
        known = visible[p]
        if known.sum() >= 2:
            out[p] = np.interp(steps, steps[known], ndvi[p, known])
    return out


def savgol_fill(
    ndvi: np.ndarray, visible: np.ndarray, window: int = 7, order: int = 2
) -> np.ndarray:
    """Linear fill, then a Savitzky-Golay filter (7 steps = 35 days) that smooths noisy values."""
    filled = linear_fill(ndvi, visible)
    out = np.full(ndvi.shape, np.nan)
    ok = ~np.isnan(filled).any(axis=1)
    out[ok] = savgol_filter(filled[ok], window_length=window, polyorder=order, axis=1)
    return out


# --- Tabular features (XGBoost) ----------------------------------------------------------------


def tabular(
    grid: Grid, rows: np.ndarray, hidden: np.ndarray, require_next: bool = True
) -> pd.DataFrame:
    """One row per hidden observation of the given plots, with what is known around it.

    Radar at the target step, the visible NDVI before and after it, how far they are, the radar at
    those two dates and its change since. The linear interpolation is included as a feature.
    """
    visible = grid.observed & ~hidden
    prev, nxt = neighbours(visible)
    linear = linear_fill(grid.ndvi, visible)
    gaps = gap_days(hidden, visible)
    doy = grid.doy
    records = []
    for p in rows:
        for t in np.flatnonzero(hidden[p]):
            tp, tn = prev[p, t], nxt[p, t]
            if tp < 0 or (tn < 0 and require_next):
                continue
            if tn < 0:  # real time: nothing visible after t, the "next" features are missing
                tn_values = (np.nan, np.nan, np.full(3, np.nan))
            else:
                tn_values = (grid.ndvi[p, tn], (tn - t) * STEP_DAYS, grid.s1[p, tn])
            s1_t, s1_p, s1_n = grid.s1[p, t], grid.s1[p, tp], tn_values[2]
            records.append(
                {
                    "parcel_id": grid.parcel_ids[p],
                    "crop": grid.crops[p],
                    "step": t,
                    "gap_days": gaps[p, t],
                    "target": grid.ndvi[p, t],
                    "rapeseed": float(grid.crops[p] == "Winter rapeseed"),
                    "doy_sin": np.sin(2 * np.pi * doy[t] / 365),
                    "doy_cos": np.cos(2 * np.pi * doy[t] / 365),
                    "vv": s1_t[0],
                    "vh": s1_t[1],
                    "ratio": s1_t[2],
                    "prev_ndvi": grid.ndvi[p, tp],
                    "next_ndvi": tn_values[0],
                    "days_since_prev": (t - tp) * STEP_DAYS,
                    "days_to_next": tn_values[1],
                    "linear": linear[p, t] if tn >= 0 else np.nan,
                    "d_vh_prev": s1_t[1] - s1_p[1],
                    "d_vh_next": s1_t[1] - s1_n[1],
                    "d_ratio_prev": s1_t[2] - s1_p[2],
                    "d_ratio_next": s1_t[2] - s1_n[2],
                }
            )
    return pd.DataFrame(records)


TABULAR_FEATURES = [
    "rapeseed", "doy_sin", "doy_cos", "vv", "vh", "ratio", "prev_ndvi", "next_ndvi",
    "days_since_prev", "days_to_next", "linear", "d_vh_prev", "d_vh_next", "d_ratio_prev",
    "d_ratio_next",
]  # fmt: skip
RADAR_FEATURES = ["vv", "vh", "ratio", "d_vh_prev", "d_vh_next", "d_ratio_prev", "d_ratio_next"]
NO_RADAR_FEATURES = [f for f in TABULAR_FEATURES if f not in RADAR_FEATURES]
# Real time: only the past is known.
PAST_NO_RADAR_FEATURES = ["rapeseed", "doy_sin", "doy_cos", "prev_ndvi", "days_since_prev"]
PAST_RADAR_FEATURES = PAST_NO_RADAR_FEATURES + ["vv", "vh", "ratio", "d_vh_prev", "d_ratio_prev"]
# The day of the year lets a model learn one season's regional NDVI curve, which does not carry over
# to the next season (notebook 5, section 12).
DATE_FEATURES = ["doy_sin", "doy_cos"]


def without_date(features: list[str]) -> list[str]:
    return [f for f in features if f not in DATE_FEATURES]


# Inputs of the models the API serves, per (mode, radar). The date is dropped wherever radar can
# replace it; in real time without radar, it is the only clue to the crop's stage (notebook 5, §13).
SERVED_FEATURES = {
    ("after_season", True): without_date(TABULAR_FEATURES),
    ("after_season", False): without_date(NO_RADAR_FEATURES),
    ("real_time", True): without_date(PAST_RADAR_FEATURES),
    ("real_time", False): PAST_NO_RADAR_FEATURES,
}


# --- Evaluation --------------------------------------------------------------------------------

GAP_BINS = [0, 15, 30, 45, 1000]
GAP_LABELS = ["≤ 15 days", "16–30 days", "31–45 days", "> 45 days"]


def errors(grid: Grid, rows: np.ndarray, hidden: np.ndarray, predictions: dict) -> pd.DataFrame:
    """One row per hidden observation of the given plots, with the absolute error of each model."""
    visible = grid.observed & ~hidden
    gaps = gap_days(hidden, visible)
    records = []
    for p in rows:
        for t in np.flatnonzero(hidden[p]):
            if gaps[p, t] < 0:
                continue
            record = {
                "parcel_id": grid.parcel_ids[p],
                "crop": grid.crops[p],
                "date": grid.dates[t],
                "gap_days": gaps[p, t],
                "target": grid.ndvi[p, t],
            }
            for name, pred in predictions.items():
                record[name] = abs(pred[p, t] - grid.ndvi[p, t])
            records.append(record)
    df = pd.DataFrame(records)
    df["gap"] = pd.cut(df["gap_days"], GAP_BINS, labels=GAP_LABELS)
    return df


def bootstrap_mae(
    errors: pd.DataFrame, column: str, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % interval of the MAE, resampling whole plots (errors within one plot are correlated)."""
    rng = np.random.default_rng(seed)
    per_plot = errors.groupby("parcel_id")[column].agg(["sum", "count"])
    sums, counts = per_plot["sum"].to_numpy(), per_plot["count"].to_numpy()
    draws = rng.integers(0, len(per_plot), (n, len(per_plot)))
    maes = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(np.percentile(maes, [2.5, 97.5]))


def paired_difference(errors: pd.DataFrame, a: str, b: str, n: int = 1000, seed: int = 0) -> dict:
    """MAE of a minus MAE of b on the same hidden observations, with a 95 % interval over plots."""
    diff = errors.assign(d=errors[a] - errors[b])
    low, high = bootstrap_mae(diff, "d", n, seed)
    return {"difference": diff["d"].mean(), "low": low, "high": high}
