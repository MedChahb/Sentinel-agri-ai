"""Download and extract the RPG (Registre Parcellaire Graphique) for one region and year.

The default is Grand Est (R44), 2025 edition: crops sown in autumn 2024 and harvested in
summer 2025, which matches the Sentinel season used in part 2.

Usage: python scripts/download_rpg.py [--region R44] [--year 2025]
"""

import argparse
import shutil
import urllib.request

import py7zr

from sentinel_agri.config import RPG_DIR

BASE_URL = "https://data.geopf.fr/telechargement/download/RPG"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="R44")
    parser.add_argument("--year", default="2025")
    args = parser.parse_args()

    name = f"RPG_4-0__GPKG_LAMB93_{args.region}_{args.year}-01-01"
    archive = RPG_DIR / f"{name}.7z"
    RPG_DIR.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"Downloading {name}.7z (about 800 MB) ...")
        # The IGN server rejects the default "Python-urllib" User-Agent with a 403.
        request = urllib.request.Request(
            f"{BASE_URL}/{name}/{name}.7z", headers={"User-Agent": "sentinel-agri"}
        )
        partial = archive.with_suffix(".7z.part")
        with urllib.request.urlopen(request) as response, open(partial, "wb") as f:
            shutil.copyfileobj(response, f, length=1 << 20)
        partial.rename(archive)

    print("Extracting ...")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(RPG_DIR)
    for gpkg in sorted(RPG_DIR.rglob("*.gpkg")):
        print(gpkg.relative_to(RPG_DIR))


if __name__ == "__main__":
    main()
