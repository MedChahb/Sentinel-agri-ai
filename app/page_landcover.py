"""Land-cover map of the Reims area in four seasons, and classification of uploaded tiles."""

import io
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tifffile
from common import DATA_DIR, GROUP_COLOURS, SAMPLES_DIR, ApiError, call_api, hex_to_rgb, legend
from PIL import Image

CLOUDY = 0.3  # share of cloudy pixels above which a patch is greyed out
SEASON_LABELS = {"winter": "Winter", "spring": "Spring", "summer": "Summer", "autumn": "Autumn"}


@st.cache_data
def landcover() -> tuple[dict, dict]:
    meta = json.loads((DATA_DIR / "landcover" / "marne.json").read_text())
    arrays = dict(np.load(DATA_DIR / "landcover" / "marne.npz"))
    return meta, arrays


@st.cache_data
def overlay(season_index: int, opacity: float, source: str) -> Image.Image:
    """The season's true-colour image with each 640 m patch tinted by its class group."""
    meta, a = landcover()
    season = meta["seasons"][season_index]
    base = Image.open(DATA_DIR / "landcover" / f"marne_{season}.jpg").convert("RGB")
    n = a["classes"].shape[1]
    groups = meta["groups"]
    if source == "model":
        names = [meta["class_to_group"][c] for c in meta["classes"]]
        group_index = np.vectorize(lambda k: groups.index(names[k]))(a["classes"][season_index])
    else:
        group_index = a["worldcover"]
    colours = np.array([hex_to_rgb(GROUP_COLOURS[g]) for g in groups], dtype=np.uint8)
    tint = colours[group_index]
    alpha = np.full((n, n), int(255 * opacity), np.uint8)
    if source == "model":
        cloudy = a["cloudy"][season_index] > CLOUDY
        tint[cloudy] = 255
        alpha[cloudy] = 200
    layer = Image.fromarray(np.dstack([tint, alpha]), "RGBA").resize(
        base.size, Image.Resampling.NEAREST
    )
    return Image.alpha_composite(base.convert("RGBA"), layer)


def agreement(season_index: int) -> tuple[float, int]:
    """Share of clear, pure patches where the model's group matches WorldCover's."""
    meta, a = landcover()
    names = [meta["class_to_group"][c] for c in meta["classes"]]
    model = np.vectorize(lambda k: meta["groups"].index(names[k]))(a["classes"][season_index])
    usable = (a["cloudy"][season_index] <= CLOUDY) & (a["worldcover_share"] >= 0.7)
    return float((model == a["worldcover"])[usable].mean()), int(usable.sum())


def rgb_preview(data: bytes) -> np.ndarray | None:
    """True-colour preview of a 13-band EuroSAT-style tile (B04, B03, B02 = bands 4, 3, 2)."""
    try:
        tile = tifffile.imread(io.BytesIO(data))
    except Exception:
        return None
    if tile.ndim != 3:
        return None
    if tile.shape[-1] == 13 and tile.shape[0] != 13:
        tile = np.moveaxis(tile, -1, 0)
    if tile.shape[0] < 4:
        return None
    rgb = np.stack([tile[3], tile[2], tile[1]], axis=-1).astype(float)
    return (np.clip((rgb - 600) / 1800, 0, 1) ** 0.8 * 255).astype(np.uint8)


def pct(p: float) -> str:
    """A probability as a percentage; two decimals above 99 %, so that 99.95 % never reads 100 %."""
    return f"{p:.2%}" if 0.99 < p < 1 else f"{p:.1%}"


@st.cache_data(show_spinner=False)
def classify_tile(name: str, data: bytes) -> dict:
    """One API call per distinct tile: picking the same sample again reuses the answer."""
    return call_api("/classify", files={"file": (name, data)})


def classify_section() -> None:
    st.subheader("Classify a tile")
    st.markdown(
        "The model takes a 64 × 64 Sentinel-2 **Level-1C** tile with all 13 bands, as a GeoTIFF "
        "(EuroSAT band order). The samples are tiles it never saw during training, with their true "
        "class, picked to show its whole range: sure and right, unsure, and **confidently wrong**. "
        "A Level-2A tile is refused on purpose: the model learned from Level-1C and gets half of "
        "Level-2A tiles wrong, with confidence."
    )
    samples = json.loads((SAMPLES_DIR / "samples.json").read_text())
    c1, c2 = st.columns([1, 1])
    choice = c1.selectbox(
        "Sample tile",
        list(samples),
        format_func=lambda n: f"{samples[n]['title']} · true class: {samples[n]['truth']}",
    )
    upload = c2.file_uploader("…or upload your own (.tif)", type=["tif", "tiff"])
    data = upload.getvalue() if upload else (SAMPLES_DIR / choice).read_bytes()
    name = upload.name if upload else choice
    truth = None if upload else samples[choice]["truth"]

    left, right = st.columns([1, 2])
    preview = rgb_preview(data)
    if preview is not None:
        left.image(
            Image.fromarray(preview).resize((256, 256), Image.Resampling.NEAREST),
            caption=f"{name} (natural colours)",
        )
    with (
        right,
        st.spinner(
            "Calling the model on AWS. After a quiet period, it takes about 10 s to wake up."
        ),
    ):
        try:
            answer = classify_tile(name, data)
        except ApiError as err:
            if err.refused:
                right.warning(f"Refused by the model: {err}")
            else:
                right.error(str(err))
            return
    probs = dict(sorted(answer["probabilities"].items(), key=lambda kv: kv[1]))
    meta, _ = landcover()
    groups = meta["class_to_group"]
    verdict = ""
    if truth is not None:
        right_group = groups[answer["label"]] == groups[truth]
        verdict = f" True class: **{truth}**, so the model is " + (
            "**right**." if right_group else "**wrong**."
        )
    message = f"**{answer['label']}**, probability {pct(answer['probability'])}.{verdict}"
    if truth is not None and not right_group:
        right.error(message)
    else:
        right.success(message)
    fig = go.Figure(
        go.Bar(
            x=list(probs.values()),
            y=list(probs),
            orientation="h",
            marker_color="#0072B2",
            text=[pct(v) for v in probs.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        xaxis_range=[0, 1.15],
        xaxis_tickformat=".0%",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    right.plotly_chart(fig)
    right.caption(
        "A high probability is not a guarantee: on real 2025 tiles around Reims, the model is "
        "right on 92 % of them, yet some of its mistakes come with more than 99 % confidence."
    )


def render() -> None:
    meta, _ = landcover()
    st.title("Land-cover map")
    st.markdown(
        f"**{meta['region']}**, about 24 × 24 km, cut into {meta['patch_m']} m tiles and "
        "classified by the land-cover model on four Level-1C images of 2025. Next to it, ESA "
        "WorldCover, an independent map made from a full year of images, for comparison."
    )
    c1, c2 = st.columns([2, 1])
    season_index = c1.radio(
        "Season",
        range(len(meta["seasons"])),
        horizontal=True,
        format_func=lambda i: f"{SEASON_LABELS[meta['seasons'][i]]} ({meta['dates'][i]})",
    )
    opacity = c2.slider("Colour strength", 0.0, 1.0, 0.45, 0.05)

    left, right = st.columns(2)
    left.image(
        overlay(season_index, opacity, "model"),
        caption="The model, on this day's image",
        width="stretch",
    )
    right.image(
        overlay(season_index, opacity, "worldcover"),
        caption="ESA WorldCover 2021 (majority class of each tile)",
        width="stretch",
    )
    legend(
        [(g, GROUP_COLOURS[g]) for g in meta["groups"]] + [("cloud (not classified)", "#ffffff")]
    )

    share, n = agreement(season_index)
    st.metric(
        "Tiles where the model agrees with WorldCover",
        f"{share:.0%}",
        help=f"Over the {n} cloud-free tiles that WorldCover sees as mostly one class.",
    )
    st.caption(
        "The model works at five groups here: its ten EuroSAT classes are merged (for example, "
        "annual and permanent crops into Crop). WorldCover is not the truth either: it calls "
        "much farmland grassland, and it is from 2021. Bare fields in winter and vineyards on the "
        "Montagne de Reims are the classic sources of disagreement."
    )
    st.divider()
    classify_section()
