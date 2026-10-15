"""Project paths and constants shared by notebooks, scripts and the API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

EUROSAT_DIR = RAW_DIR / "eurosat"
RPG_DIR = RAW_DIR / "rpg"

# Part 1
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
# Order of the bands in the EuroSAT GeoTIFFs: B8A is stored last, not after B08.
EUROSAT_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
]
# Central wavelength (nm, Sentinel-2A) and native resolution (m) of each band.
# EuroSAT resamples every band to 10 m.
BAND_WAVELENGTH_NM = {
    "B01": 443, "B02": 492, "B03": 560, "B04": 665, "B05": 704, "B06": 741, "B07": 783,
    "B08": 833, "B8A": 865, "B09": 945, "B10": 1374, "B11": 1614, "B12": 2202,
}  # fmt: skip
BAND_RESOLUTION_M = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20,
    "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}  # fmt: skip
# Wavelength order. It is also the order the SSL4EO (MoCo) pretrained weights expect.
BANDS_BY_WAVELENGTH = sorted(EUROSAT_BANDS, key=BAND_WAVELENGTH_NM.get)
RGB_BANDS = ["B04", "B03", "B02"]  # red, green, blue

# Part 2
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
RPG_CRS = "EPSG:2154"  # Lambert 93
S2_CRS = "EPSG:32631"  # UTM zone 31N, northern France
S2_OFFSET = 1000  # added to L2A reflectances since processing baseline 04.00 (2022)
SCL_MASKED = [0, 3, 8, 9, 10]  # no data, cloud shadow, cloud medium/high, cirrus
# Part 2: growing seasons of the winter crops (sowing to after harvest), and the RPG edition that
# declares the crops harvested in each season.
SEASONS = {"2023-24": ("2023-10-01", "2024-08-31"), "2024-25": ("2024-10-01", "2025-08-31")}
RPG_YEAR = {"2023-24": 2024, "2024-25": 2025}

# Experiment tracking (local MLflow server, see docker-compose.yml)
MLFLOW_TRACKING_URI = "http://localhost:5000"
