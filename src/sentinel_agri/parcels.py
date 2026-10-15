"""Part 2: the farm plots whose NDVI time series are gap-filled."""

import geopandas as gpd

from sentinel_agri.config import PROCESSED_DIR, RPG_DIR, RPG_YEAR

# 20 km x 20 km of the chalk plain east of Reims, in Lambert 93 (the RPG's projection).
AREA_L93 = (780000, 6880000, 800000, 6900000)
CRS_UTM = "EPSG:32631"
CROPS = {"BTH": "Winter wheat", "CZH": "Winter rapeseed"}


def parcels_path(season: str):
    return PROCESSED_DIR / f"part2_parcels_{season}.gpkg"


def rpg_file(season: str):
    """The RPG plot file of the year in which the season's crops are harvested."""
    year = RPG_YEAR[season]
    matches = [p for p in RPG_DIR.rglob("RPG_Parcelles.gpkg") if f"_{year}-01-01" in str(p)]
    if not matches:
        raise FileNotFoundError(
            f"RPG {year} is missing: run python scripts/download_rpg.py --year {year}"
        )
    return matches[0]


def select_parcels(
    season: str,
    n_per_crop: int | None = None,
    min_area_ha: float = 1.0,
    buffer_m: float = 10.0,
    min_pixels: int = 50,
    seed: int = 0,
    recompute: bool = False,
) -> gpd.GeoDataFrame:
    """Winter wheat and winter rapeseed plots of the area in one season, shrunk by `buffer_m`.

    All plots by default, or `n_per_crop` random ones per crop. The inward buffer removes edge
    pixels that mix the plot with its neighbours; plots left with fewer than `min_pixels` pixels
    are dropped. Cached per season.
    """
    path = parcels_path(season)
    if path.exists() and not recompute:
        return gpd.read_file(path)

    plots = gpd.read_file(
        rpg_file(season), bbox=AREA_L93, columns=["id_parcel", "code_cultu", "surf_parc"]
    )
    plots = plots[plots["code_cultu"].isin(CROPS) & (plots["surf_parc"] >= min_area_ha)]
    plots = plots.to_crs(CRS_UTM)
    plots["geometry"] = plots.buffer(-buffer_m)
    plots = plots[~plots.is_empty & (plots.area >= min_pixels * 100)]
    if n_per_crop is not None:
        plots = plots.groupby("code_cultu", group_keys=False).sample(
            n=n_per_crop, random_state=seed
        )
    plots = plots.sort_values(["code_cultu", "id_parcel"]).reset_index(drop=True)

    parcels = gpd.GeoDataFrame(
        {
            "parcel_id": range(1, len(plots) + 1),
            "rpg_id": plots["id_parcel"].to_numpy(),
            "crop": plots["code_cultu"].map(CROPS).to_numpy(),
            "area_ha": (plots.area / 10_000).round(2).to_numpy(),
            "season": season,
        },
        geometry=plots.geometry.to_numpy(),
        crs=CRS_UTM,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    parcels.to_file(path)
    return parcels
