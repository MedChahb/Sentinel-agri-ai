"""Download the 13-band EuroSAT dataset (about 2 GB) with its official splits.

Usage: python scripts/download_eurosat.py
"""

from torchgeo.datasets import EuroSAT

from sentinel_agri.config import EUROSAT_DIR


def main() -> None:
    EUROSAT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ds = EuroSAT(root=EUROSAT_DIR, split=split, download=True, checksum=True)
        print(f"{split}: {len(ds)} tiles")


if __name__ == "__main__":
    main()
