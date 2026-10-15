"""Select the plots of a season and download their NDVI and radar time series (part 2).

Every acquisition is cached, so an interrupted run resumes where it stopped.

Usage: python scripts/extract_part2.py --season 2024-25
"""

import argparse
import logging
import warnings

from sentinel_agri.config import SEASONS
from sentinel_agri.parcels import select_parcels
from sentinel_agri.timeseries import s1_backscatter, s2_ndvi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", choices=list(SEASONS), required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message=".*no geotransform.*")
    logging.getLogger("rasterio").setLevel(logging.ERROR)

    parcels = select_parcels(args.season)
    print(f"{args.season}: {len(parcels)} plots", parcels["crop"].value_counts().to_dict())
    ndvi = s2_ndvi(parcels, args.season)
    print(f"Sentinel-2: {ndvi['date'].nunique()} dates with clear plots")
    s1 = s1_backscatter(parcels, args.season)
    print(f"Sentinel-1: {s1.groupby(['date', 'orbit']).ngroups} passes")


if __name__ == "__main__":
    main()
