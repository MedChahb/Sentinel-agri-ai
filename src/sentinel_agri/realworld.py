"""Real-world test of the part 1 model: 64 x 64 patches cut from new Sentinel-2 images.

The pipeline, for one region:

1. `label_grid`: a 10 m map of true classes, from the RPG (French farm plots) and ESA WorldCover.
2. `pure_patches`: the 64 x 64 patches mostly covered by one class (see MAIN_SHARE and OTHER_SHARE).
3. `best_dates`: the least cloudy acquisition of each season of 2025.
4. `load_l1c` (Copernicus Data Space) and `load_l2a` (Planetary Computer): the 13 bands of one date,
   on the same grid as the labels, in EuroSAT file order and on EuroSAT's scale (offset removed).
5. `classify`: send each patch to the API, as any user of the model would.
"""

import io
import os
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox
from odc.stac import load
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import unary_union

from sentinel_agri.config import EUROSAT_BANDS, EUROSAT_CLASSES, PC_STAC_URL, ROOT, RPG_DIR

PATCH = 64  # pixels
RES = 10  # metres
PATCH_M = PATCH * RES
# A patch is kept when its main class covers at least MAIN_SHARE of it and every other known class
# together at most OTHER_SHARE. The rest is unmapped: roads, hedges and field borders, as in real
# tiles.
MAIN_SHARE = 0.7
OTHER_SHARE = 0.1
EUROSAT_B10_MEAN = 12  # Level-2A has no B10: it is filled with its EuroSAT mean

SEASONS = {
    "winter": ("2025-01-01", "2025-02-28"),
    "spring": ("2025-04-01", "2025-05-31"),
    "summer": ("2025-06-01", "2025-07-31"),
    "autumn": ("2025-09-01", "2025-10-31"),
}

# Scene classification (SCL) values of Level-2A that make a pixel unusable.
SCL_BAD = [0, 3, 8, 9, 10]  # no data, cloud shadow, cloud medium/high probability, thin cirrus
SCL_SNOW = 11
MIN_COVERAGE = 0.99  # share of the region the images of a date must cover

# Coarse groups: the level at which every region and EuroSAT itself can be compared.
GROUPS = ["Crop", "Grass", "Forest", "Water", "Built-up"]
EUROSAT_TO_GROUP = {
    "AnnualCrop": "Crop",
    "PermanentCrop": "Crop",
    "Pasture": "Grass",
    "HerbaceousVegetation": "Grass",
    "Forest": "Forest",
    "River": "Water",
    "SeaLake": "Water",
    "Residential": "Built-up",
    "Industrial": "Built-up",
    "Highway": "Built-up",
}

# Classes of the label maps. France has finer agricultural classes thanks to the RPG.
LABELS = ["AnnualCrop", "PermanentCrop", "Pasture", "Crop", "Grass", "Forest", "Water", "Built-up"]
LABEL_TO_GROUP = {
    "AnnualCrop": "Crop",
    "PermanentCrop": "Crop",
    "Pasture": "Grass",
    "Crop": "Crop",
    "Grass": "Grass",
    "Forest": "Forest",
    "Water": "Water",
    "Built-up": "Built-up",
}
UNLABELLED = -1

# RPG crop groups (code_group) -> label. Groups left out are ambiguous: fallow, fodder, temporary
# grassland, heath, miscellaneous.
RPG_GROUPS = {
    **{g: "AnnualCrop" for g in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 24, 25)},
    **{g: "PermanentCrop" for g in (20, 21, 22, 23)},  # orchards, vineyards, nuts, olives
    18: "Pasture",  # permanent grassland
}
# ESA WorldCover 2021 class -> label.
WORLDCOVER = {10: "Forest", 30: "Grass", 40: "Crop", 50: "Built-up", 80: "Water"}

CACHE_DIR = ROOT / "data" / "interim" / "realworld"


@dataclass(frozen=True)
class Region:
    name: str
    lon: float
    lat: float
    n_patches: int  # the region is a square of n_patches x n_patches patches
    epsg: int  # UTM zone
    labels: str  # "france": RPG + WorldCover, "worldcover": WorldCover only

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """UTM bounds, snapped to whole patches around the centre."""
        x, y = Transformer.from_crs(4326, self.epsg, always_xy=True).transform(self.lon, self.lat)
        half = self.n_patches * PATCH_M / 2
        x0, y0 = round((x - half) / PATCH_M) * PATCH_M, round((y - half) / PATCH_M) * PATCH_M
        return (x0, y0, x0 + self.n_patches * PATCH_M, y0 + self.n_patches * PATCH_M)

    @property
    def bbox_lonlat(self) -> list[float]:
        to_ll = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        x0, y0, x1, y1 = self.bbox
        lons, lats = to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        return [min(lons), min(lats), max(lons), max(lats)]

    @property
    def geobox(self) -> GeoBox:
        return GeoBox.from_bbox(self.bbox, crs=f"EPSG:{self.epsg}", resolution=RES)


REGIONS = {
    "marne": Region("Marne: Reims and the Montagne de Reims", 4.03, 49.22, 37, 32631, "france"),
    "lac_du_der": Region("Marne: Lac du Der", 4.76, 48.57, 12, 32631, "france"),
    "morocco": Region("Morocco: Saïss plain near Meknès", -5.45, 33.93, 37, 32630, "worldcover"),
}


# --- Ground truth -------------------------------------------------------------------------------


def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def worldcover(region: Region) -> np.ndarray:
    """ESA WorldCover 2021 class of each 10 m pixel of the region."""
    items = (
        _catalog().search(collections=["esa-worldcover"], bbox=region.bbox_lonlat).item_collection()
    )
    items = [i for i in items if "2021" in i.id]
    return load(items, bands=["map"], geobox=region.geobox, resampling="nearest")["map"].values[0]


def rpg(region: Region) -> np.ndarray:
    """Label index of each pixel from the RPG 2025, UNLABELLED outside declared plots.

    Plots of ambiguous crop groups are burnt in as -2, so that a patch containing them is impure.
    """
    path = next(RPG_DIR.rglob("RPG_Parcelles.gpkg"))
    box = gpd.GeoSeries.from_xy([region.bbox[0], region.bbox[2]], [region.bbox[1], region.bbox[3]])
    box = box.set_crs(region.epsg).to_crs(2154)
    plots = gpd.read_file(path, bbox=tuple(box.total_bounds), columns=["code_group"]).to_crs(
        region.epsg
    )
    codes = (
        plots["code_group"]
        .astype(int)
        .map(RPG_GROUPS)
        .map(lambda c: LABELS.index(c) if isinstance(c, str) else -2)
    )
    gb = region.geobox
    return rasterize(
        zip(plots.geometry, codes, strict=True),
        out_shape=gb.shape,
        transform=gb.affine,
        fill=UNLABELLED,
        dtype="int16",
    )


def label_grid(region: Region) -> np.ndarray:
    """Label index (into LABELS) of each 10 m pixel; negative where unknown or ambiguous."""
    wc = worldcover(region)
    grid = np.full(wc.shape, UNLABELLED, dtype=np.int16)
    for value, label in WORLDCOVER.items():
        grid[wc == value] = LABELS.index(label)
    if region.labels == "france":
        # Farmland comes from the RPG only: cropland or grassland outside declared plots is unknown.
        grid[np.isin(grid, [LABELS.index("Crop"), LABELS.index("Grass")])] = UNLABELLED
        plots = rpg(region)
        inside = plots != UNLABELLED
        grid[inside] = plots[inside]
    return grid


def pure_patches(grid: np.ndarray) -> pd.DataFrame:
    """Every patch of the regular 64 x 64 grid dominated by one label (MAIN_SHARE, OTHER_SHARE)."""
    rows = []
    for r in range(grid.shape[0] // PATCH):
        for c in range(grid.shape[1] // PATCH):
            window = grid[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH].ravel()
            values, counts = np.unique(window[window != UNLABELLED], return_counts=True)
            if not len(values):
                continue
            main = counts.argmax()
            main_share = counts[main] / window.size
            other_share = (counts.sum() - counts[main]) / window.size
            if values[main] >= 0 and main_share >= MAIN_SHARE and other_share <= OTHER_SHARE:
                rows.append(
                    {"row": r, "col": c, "label": LABELS[values[main]], "main_share": main_share}
                )
    df = pd.DataFrame(rows)
    df["group"] = df["label"].map(LABEL_TO_GROUP)
    return df


def sample_patches(patches: pd.DataFrame, n_per_label: int = 100, seed: int = 0) -> pd.DataFrame:
    """Up to `n_per_label` random patches of each label."""
    samples = [
        group.sample(min(len(group), n_per_label), random_state=seed)
        for _, group in patches.groupby("label")
    ]
    return pd.concat(samples, ignore_index=True)


# --- Imagery ------------------------------------------------------------------------------------


def best_dates(region: Region) -> pd.DataFrame:
    """For each season, the least cloudy acquisition date whose images cover the whole region.

    Without the coverage check, a date where the region sits at the edge of the satellite's swath
    can look cloud-free while most of the region has no data.
    """
    cat = _catalog()
    region_box = box(*region.bbox_lonlat)
    rows = []
    for season, (start, end) in SEASONS.items():
        items = cat.search(
            collections=["sentinel-2-l2a"], bbox=region.bbox_lonlat, datetime=f"{start}/{end}"
        ).item_collection()
        df = pd.DataFrame(
            {
                "date": [i.datetime.date().isoformat() for i in items],
                "cloud": [i.properties["eo:cloud_cover"] for i in items],
                "footprint": [shape(i.geometry) for i in items],
            }
        )
        per_date = df.groupby("date").agg(
            cloud=("cloud", "mean"), footprint=("footprint", lambda f: unary_union(list(f)))
        )
        per_date["coverage"] = per_date["footprint"].map(
            lambda g: g.intersection(region_box).area / region_box.area
        )
        candidates = per_date[per_date["coverage"] >= MIN_COVERAGE].sort_values("cloud")
        best = candidates.iloc[0]
        rows.append(
            {
                "season": season,
                "date": candidates.index[0],
                "scene_cloud_%": round(best["cloud"], 1),
                "coverage_%": round(100 * best["coverage"], 1),
            }
        )
    return pd.DataFrame(rows)


def load_l2a(region: Region, date: str) -> tuple[np.ndarray, np.ndarray]:
    """Level-2A bands of one date (EuroSAT order, offset removed, B10 filled) and the SCL layer."""
    items = (
        _catalog()
        .search(collections=["sentinel-2-l2a"], bbox=region.bbox_lonlat, datetime=f"{date}/{date}")
        .item_collection()
    )
    bands = [b for b in EUROSAT_BANDS if b != "B10"]
    ds = load(
        items, bands=bands + ["SCL"], geobox=region.geobox, groupby="solar_day",
        resampling={"*": "bilinear", "SCL": "nearest"},
    )  # fmt: skip
    offset = 1000 if float(items[0].properties["s2:processing_baseline"]) >= 4 else 0
    stack = []
    for b in EUROSAT_BANDS:
        if b == "B10":
            stack.append(np.full(region.geobox.shape, EUROSAT_B10_MEAN, dtype=np.float32))
        else:
            values = np.nan_to_num(ds[b].values[0].astype(np.float32), nan=0)  # no data -> 0
            stack.append(np.clip(values - offset, 0, None))
    scl = np.nan_to_num(ds["SCL"].values[0], nan=0).astype(np.uint8)
    return np.stack(stack).astype(np.uint16), scl


def _env(key: str) -> str:
    if key in os.environ:
        return os.environ[key]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise KeyError(f"{key} is missing from .env")


def _cdse_token() -> str:
    response = httpx.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": _env("CDSE_CLIENT_ID"),
            "client_secret": _env("CDSE_CLIENT_SECRET"),
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def load_l1c(region: Region, date: str) -> np.ndarray:
    """Level-1C bands of one date from Copernicus Data Space, in EuroSAT order and scale.

    `harmonizeValues` (on by default) removes the +1,000 offset of recent processing baselines.
    The 20 m and 60 m bands are resampled bilinearly, as in EuroSAT.
    """
    evalscript = f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: {EUROSAT_BANDS}, units: "DN" }}],
    output: {{ bands: 13, sampleType: "UINT16" }}
  }};
}}
function evaluatePixel(s) {{ return [{", ".join("s." + b for b in EUROSAT_BANDS)}]; }}"""
    width, height = region.geobox.shape.x, region.geobox.shape.y
    body = {
        "input": {
            "bounds": {
                "bbox": list(region.bbox),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{region.epsg}"},
            },
            "data": [
                {
                    "type": "sentinel-2-l1c",
                    "dataFilter": {
                        "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                        "mosaickingOrder": "mostRecent",
                    },
                    "processing": {"upsampling": "BILINEAR"},
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    response = httpx.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        json=body,
        headers={"Authorization": f"Bearer {_cdse_token()}"},
        timeout=600,
    )
    response.raise_for_status()
    with rasterio.open(io.BytesIO(response.content)) as src:
        return src.read()


def cut(image: np.ndarray, patches: pd.DataFrame) -> np.ndarray:
    """(n, bands, 64, 64) array of the given patches."""
    return np.stack(
        [
            image[..., r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
            for r, c in zip(patches.row, patches.col, strict=True)
        ]
    )


def patch_quality(scl: np.ndarray, patches: pd.DataFrame) -> pd.DataFrame:
    """Share of cloudy / missing pixels and of snow pixels in each patch, from the Level-2A SCL."""
    windows = cut(scl, patches).reshape(len(patches), -1)
    return pd.DataFrame(
        {
            "bad_share": np.isin(windows, SCL_BAD).mean(axis=1),
            "snow_share": (windows == SCL_SNOW).mean(axis=1),
        }
    )


def load_or_download(
    region_key: str, season: str, date: str, patches: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Patches of one region and date, for both products, cached in data/interim/realworld/."""
    path = CACHE_DIR / f"{region_key}_{season}_{date}.npz"
    if path.exists():
        return dict(np.load(path))
    region = REGIONS[region_key]
    l2a, scl = load_l2a(region, date)
    l1c = load_l1c(region, date)
    quality = patch_quality(scl, patches)
    data = {
        "l1c": cut(l1c, patches),
        "l2a": cut(l2a, patches),
        "bad_share": quality.bad_share.to_numpy(),
        "snow_share": quality.snow_share.to_numpy(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


# --- Predictions --------------------------------------------------------------------------------


def to_geotiff(patch: np.ndarray) -> bytes:
    with MemoryFile() as memfile:
        # The API ignores the georeference; a dummy one avoids a "not georeferenced" warning.
        with memfile.open(
            driver="GTiff",
            count=patch.shape[0],
            height=PATCH,
            width=PATCH,
            dtype="uint16",
            transform=from_origin(0, PATCH * RES, RES, RES),
        ) as dst:
            dst.write(patch.astype(np.uint16))
        return memfile.read()


def classify(patches: np.ndarray, api_url: str = "http://localhost:8000") -> np.ndarray:
    """(n, 10) class probabilities from the API, in EUROSAT_CLASSES order."""
    probs = []
    with httpx.Client(base_url=api_url, timeout=30) as client:
        for patch in patches:
            response = client.post("/classify", files={"file": ("patch.tif", to_geotiff(patch))})
            response.raise_for_status()
            probabilities = response.json()["probabilities"]
            probs.append([probabilities[c] for c in EUROSAT_CLASSES])
    return np.array(probs)


def classify_files(paths: list[str], api_url: str = "http://localhost:8000") -> np.ndarray:
    """(n, 10) class probabilities from the API for GeoTIFF files, such as EuroSAT tiles."""
    probs = []
    with httpx.Client(base_url=api_url, timeout=30) as client:
        for path in paths:
            tile = Path(path).read_bytes()
            response = client.post("/classify", files={"file": ("tile.tif", tile)})
            response.raise_for_status()
            probabilities = response.json()["probabilities"]
            probs.append([probabilities[c] for c in EUROSAT_CLASSES])
    return np.array(probs)


def api_is_ready(api_url: str = "http://localhost:8000") -> bool:
    try:
        return httpx.get(f"{api_url}/health", timeout=5).json().get("model_loaded", False)
    except httpx.HTTPError:
        return False


def save_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name
