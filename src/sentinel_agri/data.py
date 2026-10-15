"""EuroSAT loading: datasets, data loaders and per-tile summary features."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from sentinel_agri.config import EUROSAT_BANDS, EUROSAT_DIR, PROCESSED_DIR

SPLITS = ("train", "val", "test")
FEATURES_PATH = PROCESSED_DIR / "eurosat_tile_features.parquet"


def eurosat(split: str, bands: list[str] = EUROSAT_BANDS) -> EuroSAT:
    """One EuroSAT split, with `bands` stacked in the given order.

    Each sample is {"image": float tensor (len(bands), 64, 64) of raw values, "label": int}.
    """
    return EuroSAT(root=EUROSAT_DIR, split=split, bands=tuple(bands))


def make_loaders(
    bands: list[str] = EUROSAT_BANDS, batch_size: int = 128, num_workers: int = 4
) -> dict[str, DataLoader]:
    """Train, validation and test loaders. Only the training loader is shuffled."""
    return {
        split: DataLoader(
            eurosat(split, bands),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
        for split in SPLITS
    }


def _features_of_split(split: str) -> pd.DataFrame:
    ds = eurosat(split)
    loader = DataLoader(ds, batch_size=512, num_workers=4)
    red, nir = EUROSAT_BANDS.index("B04"), EUROSAT_BANDS.index("B08")

    rows = []
    for batch in loader:
        x = batch["image"]  # (B, 13, 64, 64)
        ndvi = (x[:, nir] - x[:, red]) / (x[:, nir] + x[:, red]).clamp(min=1)
        rows.append(
            np.column_stack(
                [
                    x.mean(dim=(2, 3)).numpy(),
                    x.std(dim=(2, 3)).numpy(),
                    ndvi.mean(dim=(1, 2)).numpy(),
                    ndvi.std(dim=(1, 2)).numpy(),
                ]
            )
        )

    columns = (
        [f"mean_{b}" for b in EUROSAT_BANDS]
        + [f"std_{b}" for b in EUROSAT_BANDS]
        + ["ndvi_mean", "ndvi_std"]
    )
    df = pd.DataFrame(np.concatenate(rows), columns=columns)
    df["label"] = [label for _, label in ds.samples]
    df["class"] = df["label"].map(dict(enumerate(ds.classes)))
    df["split"] = split
    df["path"] = [path for path, _ in ds.samples]
    return df


def load_tile_features(recompute: bool = False) -> pd.DataFrame:
    """Per-tile summary of the 3 splits: mean and std of each band, NDVI mean and std.

    The first call reads all 27,000 tiles (under a minute) and caches the table to Parquet.
    """
    if FEATURES_PATH.exists() and not recompute:
        return pd.read_parquet(FEATURES_PATH)
    df = pd.concat([_features_of_split(s) for s in SPLITS], ignore_index=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH)
    return df


def band_stats(features: pd.DataFrame, bands: list[str]) -> tuple[list[float], list[float]]:
    """Mean and std of each band over all pixels of the given tiles.

    All tiles have the same size, so the pixel mean is the mean of the tile means, and the
    pixel variance is the mean within-tile variance plus the variance of the tile means.
    """
    means, stds = [], []
    for b in bands:
        tile_means, tile_stds = features[f"mean_{b}"], features[f"std_{b}"]
        means.append(float(tile_means.mean()))
        stds.append(float(np.sqrt((tile_stds**2).mean() + tile_means.var(ddof=0))))
    return means, stds


def to_rgb(
    image: torch.Tensor | np.ndarray,
    bands: list[str] = EUROSAT_BANDS,
    show: tuple[str, str, str] = ("B04", "B03", "B02"),
) -> np.ndarray:
    """Displayable (64, 64, 3) array in [0, 1] from a tile whose bands are in `bands` order.

    `show` picks the bands displayed as red, green and blue: true colour by default,
    ("B08", "B04", "B03") for the false-colour view where vegetation appears red.

    Reflectances are small numbers (a forest is dark), so each tile is stretched between its
    2nd and 98th percentiles to be visible. Colours are therefore not comparable across tiles.
    """
    image = np.asarray(image, dtype=np.float32)
    rgb = np.stack([image[bands.index(b)] for b in show], axis=-1)
    low, high = np.percentile(rgb, (2, 98))
    return np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
