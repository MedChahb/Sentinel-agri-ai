"""Sentinel Agri AI: web app.

Run locally:   streamlit run app/streamlit_app.py
The pages read precomputed results (scripts/build_app_data.py) and call the model API for live
predictions: the AWS Lambda one by default, or API_URL from the environment or the app's secrets.
"""

import page_fields
import page_landcover
import page_roadmap
import page_try
import streamlit as st
from common import REPO_URL, api_url, wake_api

st.set_page_config(
    page_title="Sentinel Agri AI", page_icon=":material/satellite_alt:", layout="wide"
)


def about() -> None:
    st.title("About")
    st.markdown(
        f"""
Sentinel Agri AI follows crops from space with free Copernicus satellite data: Sentinel-2 (optical,
blocked by clouds) and Sentinel-1 (radar, which sees through them).

**What the app does**

- **Field monitor:** how is each of 1,254 fields doing on a given day, even under clouds?
  Precomputed.
- **Try the gap-filling:** can the model rebuild a period hidden by clouds? Live call to the
  model API.
- **Land-cover map:** what covers the ground around Reims, season by season? Precomputed, plus
  live classification of your own tiles.

**How it is built**

```
Your browser  →  this Streamlit app  →  model API (FastAPI on AWS Lambda)  →  models
                 (reads precomputed results)       (ONNX classifier, XGBoost gap-filling)
```

The field and land-cover maps come from a batch job that ran the same models on every field and
tile, as a monitoring service updates its fields after each satellite pass. The live pages send
requests to the API, exactly as any other program could: its interactive documentation is at
[{api_url()}/docs]({api_url()}/docs).

**Measured accuracy** (details in the notebooks)

- Land cover: 98.98 % on EuroSAT's test set, 96.6 % on new 2025 Level-1C images around Reims,
  44.6 % on Level-2A images. The API refuses Level-2A by default; this app sends them anyway and
  shows a warning, so that you can see the model fail on them.
- Gap-filling, models trained on 2023–24 and applied to 2024–25 under real cloud patterns: 17 %
  lower error than a straight line across the gap, and in real time, 20 % lower than keeping the
  last clear value. Radar is what makes the difference: without it, no model reliably beats those
  simple answers.

**Limits**

- Winter wheat and rapeseed east of Reims only; one pair of seasons tested.
- The gap-filling is best from April to the harvest. For wheat greening up in late winter after a
  long gap, it overshoots: the 2023-24 calendar it learned was a month early.
- Typical error about 0.08 NDVI: enough to tell a field's stage, not subtle stress.
- The land-cover model needs Level-1C images, the less common Sentinel-2 product.

**Data:** ESA Copernicus Sentinel-1 and Sentinel-2 (via Microsoft Planetary Computer and the
Copernicus Data Space), EuroSAT, the French RPG (IGN), ESA WorldCover 2021.

Code, notebooks and results: [{REPO_URL}]({REPO_URL})
"""
    )


wake_api()  # start the model API while the visitor reads, so that live calls answer quickly
pages = [
    st.Page(
        page_fields.render,
        title="Field monitor",
        icon=":material/agriculture:",
        url_path="fields",
        default=True,
    ),
    st.Page(page_try.render, title="Try the gap-filling", icon=":material/cloud:", url_path="try"),
    st.Page(
        page_landcover.render, title="Land-cover map", icon=":material/map:", url_path="landcover"
    ),
    st.Page(
        page_roadmap.render,
        title="From demo to real service",
        icon=":material/rocket_launch:",
        url_path="roadmap",
    ),
    st.Page(about, title="About", icon=":material/info:", url_path="about"),
]
st.sidebar.markdown("**Sentinel Agri AI**  \nCrops from space, despite the clouds.")
st.navigation(pages).run()
st.sidebar.caption(f"[Code on GitHub]({REPO_URL}) · [Model API]({api_url()}/docs)")
