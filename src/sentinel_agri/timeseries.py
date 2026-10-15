"""Part 2: per-plot Sentinel-2 NDVI and Sentinel-1 backscatter for a growing season.

Both are read from Microsoft Planetary Computer (no account needed). Each acquisition is reduced to
one value per plot and cached in data/interim/part2/, so an interrupted download resumes.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
from odc.geo.geobox import GeoBox
from odc.stac import load
from rasterio.features import rasterize

from sentinel_agri.config import INTERIM_DIR, PC_STAC_URL, PROCESSED_DIR, SEASONS

CACHE_DIR = INTERIM_DIR / "part2"


def ndvi_path(season: str):
    return PROCESSED_DIR / f"part2_ndvi_{season}.parquet"


def s1_path(season: str):
    return PROCESSED_DIR / f"part2_s1_{season}.parquet"


# Scene classification (SCL) values that make a Sentinel-2 pixel unusable for NDVI.
SCL_UNUSABLE = [0, 1, 3, 8, 9, 10, 11]  # no data, defective, shadow, clouds, cirrus, snow
MAX_SCENE_CLOUD = 98  # dates where every tile is almost fully cloudy are skipped
S1_MIN_COVER = 0.9  # share of a plot the radar swath must cover
READ_THREADS = 8  # parallel reads of image blocks


def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def plot_raster(parcels: gpd.GeoDataFrame, resolution: int) -> tuple[GeoBox, np.ndarray]:
    """A grid covering all plots, and the parcel_id of each pixel (0 outside the plots)."""
    x0, y0, x1, y1 = parcels.total_bounds
    bounds = (
        np.floor(x0 / resolution) * resolution,
        np.floor(y0 / resolution) * resolution,
        np.ceil(x1 / resolution) * resolution,
        np.ceil(y1 / resolution) * resolution,
    )
    geobox = GeoBox.from_bbox(bounds, crs=parcels.crs, resolution=resolution)
    ids = rasterize(
        zip(parcels.geometry, parcels["parcel_id"], strict=True),
        out_shape=geobox.shape,
        transform=geobox.affine,
        fill=0,
        dtype="int32",
    )
    return geobox, ids


def zonal_mean(values: np.ndarray, valid: np.ndarray, ids: np.ndarray, n: int):
    """Mean of `values` over the valid pixels of each plot, and the valid share of each plot."""
    keep = valid & (ids > 0)
    sums = np.bincount(ids[keep], weights=values[keep], minlength=n + 1)
    counts = np.bincount(ids[keep], minlength=n + 1)
    totals = np.bincount(ids.ravel(), minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums / counts)[1:], (counts / totals)[1:]


def _bbox_lonlat(parcels: gpd.GeoDataFrame) -> list[float]:
    return list(parcels.to_crs(4326).total_bounds)


def s2_ndvi(parcels: gpd.GeoDataFrame, season: str, recompute: bool = False) -> pd.DataFrame:
    """Mean NDVI of each plot on every Sentinel-2 date, with the share of the plot that is clear.

    Columns: parcel_id, date, ndvi, clear_share. A row exists when at least one pixel is clear.
    """
    if ndvi_path(season).exists() and not recompute:
        return pd.read_parquet(ndvi_path(season))
    geobox, ids = plot_raster(parcels, 10)
    n = int(parcels["parcel_id"].max())
    items = (
        _catalog()
        .search(
            collections=["sentinel-2-l2a"],
            bbox=_bbox_lonlat(parcels),
            datetime="/".join(SEASONS[season]),
        )
        .item_collection()
    )
    by_date: dict[str, list] = {}
    for item in items:
        by_date.setdefault(item.datetime.date().isoformat(), []).append(item)

    cache = CACHE_DIR / season / "s2"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (date, date_items) in enumerate(sorted(by_date.items())):
        path = cache / f"{date}.parquet"
        if not path.exists():
            cloud = np.mean([it.properties["eo:cloud_cover"] for it in date_items])
            frame = pd.DataFrame(columns=["parcel_id", "date", "ndvi", "clear_share"])
            if cloud < MAX_SCENE_CLOUD:
                ds = load(
                    date_items, bands=["B04", "B08", "SCL"], geobox=geobox, groupby="solar_day",
                    resampling={"*": "bilinear", "SCL": "nearest"}, pool=READ_THREADS,
                )  # fmt: skip
                baseline = float(date_items[0].properties["s2:processing_baseline"])
                offset = 1000 if baseline >= 4 else 0  # added to L2A values since baseline 04.00
                red = ds["B04"].values[0].astype(np.float32) - offset
                nir = ds["B08"].values[0].astype(np.float32) - offset
                scl = np.nan_to_num(ds["SCL"].values[0], nan=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndvi = (nir - red) / (nir + red)
                valid = ~np.isin(scl, SCL_UNUSABLE) & np.isfinite(ndvi) & (red > 0) & (nir > 0)
                mean, share = zonal_mean(np.nan_to_num(ndvi), valid, ids, n)
                frame = pd.DataFrame(
                    {
                        "parcel_id": np.arange(1, n + 1),
                        "date": date,
                        "ndvi": mean,
                        "clear_share": share,
                    }
                )
                frame = frame[frame["clear_share"] > 0]
            frame.to_parquet(path)
            print(
                f"Sentinel-2 {i + 1}/{len(by_date)}: {date}, {len(frame)} plots with clear pixels"
            )
        frames.append(pd.read_parquet(path))
    ndvi = pd.concat(frames, ignore_index=True)
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    ndvi["parcel_id"] = ndvi["parcel_id"].astype(int)
    ndvi.to_parquet(ndvi_path(season))
    return ndvi


def s1_backscatter(parcels: gpd.GeoDataFrame, season: str, recompute: bool = False) -> pd.DataFrame:
    """Mean radar backscatter of each plot on every Sentinel-1 acquisition (RTC product, 20 m).

    Means are taken on linear values, then converted to decibels. Columns: parcel_id, date, orbit,
    orbit_state, vv_db, vh_db, ratio_db (VH − VV in dB, that is the VH/VV ratio).
    """
    if s1_path(season).exists() and not recompute:
        return pd.read_parquet(s1_path(season))
    geobox, ids = plot_raster(parcels, 20)
    n = int(parcels["parcel_id"].max())
    items = (
        _catalog()
        .search(
            collections=["sentinel-1-rtc"],
            bbox=_bbox_lonlat(parcels),
            datetime="/".join(SEASONS[season]),
        )
        .item_collection()
    )
    passes: dict[tuple[str, int], list] = {}  # one pass = one date and one relative orbit
    for item in items:
        key = (item.datetime.date().isoformat(), item.properties["sat:relative_orbit"])
        passes.setdefault(key, []).append(item)

    cache = CACHE_DIR / season / "s1"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, ((date, orbit), pass_items) in enumerate(sorted(passes.items())):
        path = cache / f"{date}_{orbit}.parquet"
        if not path.exists():
            # 10 m pixels averaged to 20 m: less speckle, four times less data.
            ds = load(
                pass_items, bands=["vv", "vh"], geobox=geobox, groupby="solar_day",
                resampling="average", pool=READ_THREADS,
            )  # fmt: skip
            vv, vh = ds["vv"].values[0], ds["vh"].values[0]
            valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
            vv_mean, share = zonal_mean(np.nan_to_num(vv), valid, ids, n)
            vh_mean, _ = zonal_mean(np.nan_to_num(vh), valid, ids, n)
            frame = pd.DataFrame(
                {
                    "parcel_id": np.arange(1, n + 1),
                    "date": date,
                    "orbit": int(orbit),
                    "orbit_state": pass_items[0].properties["sat:orbit_state"],
                    "vv_db": 10 * np.log10(vv_mean),
                    "vh_db": 10 * np.log10(vh_mean),
                    "cover": share,
                }
            )
            frame = frame[frame["cover"] >= S1_MIN_COVER].drop(columns="cover")
            frame.to_parquet(path)
            print(f"Sentinel-1 {i + 1}/{len(passes)}: {date}, orbit {orbit}, {len(frame)} plots")
        frames.append(pd.read_parquet(path))
    s1 = pd.concat(frames, ignore_index=True)
    s1["date"] = pd.to_datetime(s1["date"])
    s1["parcel_id"] = s1["parcel_id"].astype(int)
    s1["ratio_db"] = s1["vh_db"] - s1["vv_db"]
    s1.to_parquet(s1_path(season))
    return s1
