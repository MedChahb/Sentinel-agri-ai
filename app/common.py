"""Shared by the pages: data loading, the API client and the colour scales."""

import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLES_DIR = APP_DIR / "samples"

DEFAULT_API_URL = "https://cga6hukfpyansgulvry42cg3xi0ygxaa.lambda-url.eu-north-1.on.aws"
REPO_URL = "https://github.com/MedChahb/Sentinel-agri-ai"
STEP_DAYS = 5


def api_url() -> str:
    """The model API: API_URL from the app's secrets or environment, else the AWS Lambda one."""
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:  # no secrets file: running locally
        pass
    return os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")


# --- The model API ------------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, message: str, refused: bool = False):
        super().__init__(message)
        self.refused = refused  # the API understood the request and declined it (HTTP 422)


def wake_api() -> None:
    """Start the API in the background as soon as a visitor arrives, once per session.

    On AWS Lambda, the first request after a quiet period starts the container (about 7 s). Waking
    it while the visitor reads the first page hides most of that wait.
    """
    if not st.session_state.get("api_woken"):
        st.session_state["api_woken"] = True
        threading.Thread(target=_ping, daemon=True).start()


def _ping() -> None:
    try:
        requests.get(f"{api_url()}/health", timeout=60)
    except requests.RequestException:
        pass


def call_api(path: str, **kwargs) -> dict:
    """POST to the API. Retries once, because a cold start can fail the very first call."""
    last_error = None
    for _ in range(2):
        try:
            response = requests.post(f"{api_url()}{path}", timeout=60, **kwargs)
        except requests.RequestException as err:
            last_error = f"The model API did not answer ({type(err).__name__})."
            continue
        if response.status_code in (502, 503, 504):
            last_error = f"The model API is starting up (HTTP {response.status_code})."
            continue
        if response.status_code == 422:
            detail = response.json().get("detail")
            raise ApiError(detail if isinstance(detail, str) else json.dumps(detail), refused=True)
        if not response.ok:
            raise ApiError(f"The model API answered HTTP {response.status_code}.")
        return response.json()
    raise ApiError(f"{last_error} Please try again in a few seconds.")


# --- Data ---------------------------------------------------------------------------------------


@st.cache_data
def fields_meta() -> dict:
    meta = json.loads((DATA_DIR / "fields_meta.json").read_text())
    meta["dates"] = pd.to_datetime(meta["dates"])
    return meta


@st.cache_data
def series() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "series.parquet")


@st.cache_data
def plots() -> dict:
    return json.loads((DATA_DIR / "plots.geojson").read_text())


@st.cache_data
def plot_table() -> pd.DataFrame:
    return pd.DataFrame([f["properties"] for f in plots()["features"]]).set_index("parcel_id")


@st.cache_data
def clear_ndvi() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "clear_ndvi.parquet")


@st.cache_data
def radar() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "radar.parquet")


@st.cache_data
def matrices() -> dict[str, np.ndarray]:
    """(plots, steps) arrays of the series, in plot_table() order."""
    s = series()
    ids = plot_table().index.to_numpy()
    n_steps = len(fields_meta()["dates"])
    out = {}
    for column in ("observed", "after_season", "real_time"):
        wide = s.pivot(index="parcel_id", columns="step", values=column).reindex(ids)
        out[column] = wide.to_numpy(dtype=float)[:, :n_steps]
    return out


# --- Colours (colour-blind safe) ----------------------------------------------------------------

VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
BROWN_GREEN = ["#8c510a", "#d8b365", "#f5f5f5", "#5ab4ac", "#01665e"]  # below / above peers
YELLOW_RED = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
NO_DATA = [200, 200, 200, 160]

GROUP_COLOURS = {  # Okabe-Ito based
    "Crop": "#E69F00",
    "Grass": "#F0E442",
    "Forest": "#009E73",
    "Water": "#56B4E9",
    "Built-up": "#CC79A7",
}


def hex_to_rgb(colour: str) -> list[int]:
    return [int(colour[i : i + 2], 16) for i in (1, 3, 5)]


def colour_scale(values: np.ndarray, low: float, high: float, stops: list[str]) -> list[list[int]]:
    """RGBA colours for values between low and high, interpolated between the stops."""
    rgb = np.array([hex_to_rgb(c) for c in stops], dtype=float)
    position = np.clip((values - low) / (high - low), 0, 1) * (len(stops) - 1)
    out = []
    for p in position:
        if np.isnan(p):
            out.append(NO_DATA)
            continue
        i = min(int(p), len(stops) - 2)
        c = rgb[i] + (rgb[i + 1] - rgb[i]) * (p - i)
        out.append([int(v) for v in c] + [210])
    return out


def legend(items: list[tuple[str, str]]) -> None:
    """A row of coloured chips with labels, so that colours are always named in text."""
    chips = "".join(
        f'<span style="display:inline-block;margin-right:14px;white-space:nowrap">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{colour};'
        f'border:1px solid #888;margin-right:5px;vertical-align:middle"></span>{label}</span>'
        for label, colour in items
    )
    st.markdown(f'<div style="font-size:0.85rem">{chips}</div>', unsafe_allow_html=True)
