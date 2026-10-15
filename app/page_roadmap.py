"""From demo to real service: why the app replays a past season, and what a real service needs."""

import streamlit as st


def render() -> None:
    st.title("From demo to real service")
    st.markdown(
        "This app works on real fields and real satellite data, but it is a demo: it replays the "
        "2024–25 season. This page explains why, what a real service would add, and what was done "
        "here so that its answers can be trusted."
    )

    st.header("Why a replay, not a live service")
    st.markdown(
        """
A live service runs a **pipeline** after each satellite pass, every two to five days: download the
new Sentinel-1 and Sentinel-2 images of every field, update the time series, run the models, and
store the results in a database that the app reads. The current season's estimates are always up
to date.

```
every few days:  satellite archives  →  extraction job  →  models  →  database (PostGIS)  →  app
```

It was left out on purpose: **the rule of this project was that everything stays free, for good.**

- **The database runs all the time.** A managed PostgreSQL with PostGIS on AWS costs about $15 a
  month. The extraction itself is cheap (cents a run), but it takes about 40 minutes per season
  for these fields, more than the 15 minutes AWS Lambda allows, so it needs a container service,
  billed too.
- **AWS is only free for six months.** The Free plan's credits would cover a live pipeline for a
  while, then the account must move to the paid plan.
- **This season's fields are not public yet.** The French register of farm plots (RPG) of a season
  is published the following year. A real service gets its fields from its users instead.

The demo therefore keeps only what costs nothing or almost nothing: this app on Streamlit Community
Cloud (free), results precomputed once, and the models on AWS Lambda, within its permanent free
allowance (after the six-month Free plan, only storing the image costs a few cents a month).

**A free route exists, with limits:** scheduled GitHub Actions runs (free for public repositories,
up to six hours each) could run the extraction, and a free hosted PostgreSQL tier could store the
results. That is the natural next step.
"""
    )

    st.header("What else a real service needs")
    st.markdown(
        """
- **The user's own fields:** upload their outlines or draw them, with an account so that each user
  sees only theirs.
- **Alerts:** "3 of your fields fell behind this week", by email, instead of a list to open.
- **More crops, regions and seasons:** two crops east of Reims are a proof of concept.
- **A clock that follows the weather:** growing degree days instead of the calendar, to fix the
  known weak spot (wheat greening up after a long winter gap).
- **A land-cover model for Level-2A**, the Sentinel-2 product most users have.
- **Monitoring and retraining:** compare the estimates with each new clear image, and retrain when
  the errors grow, as the calendar shift found in this project would require every few seasons.
- **Operations:** API keys and usage limits, alarms on failures, versioned models, automatic
  deployment, and data protection (field outlines are farmers' data).
"""
    )

    st.header("What was done so that the answers can be trusted")
    st.markdown(
        """
- **Ranges, not only numbers.** Every estimate comes with a likely range (the shaded band on the
  charts). Quantile models give the 10th and 90th percentiles, then **conformal calibration** on
  held-out data widens them just enough. Without calibration, the ranges held the truth only 62–72
  % of the time in the new season; with it, 75–79 %, close to the 80 % promised. The range widens
  by itself after long gaps and in real time.
- **Tested like real use:** models trained on 2023–24 only, scored on 2024–25, under real cloud
  patterns, and the list of fields to check validated against the images that came ten days later
  (9 of 10 flagged fields were really behind).
- **Known failures detected and shown:** frost days, when the radar of every field drops at once,
  are flagged and handled; Level-2A tiles are classified only with a warning; the weak spot of the
  gap- filling is measured and displayed.
"""
    )
