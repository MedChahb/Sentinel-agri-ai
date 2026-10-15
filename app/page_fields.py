"""Field monitor: every field's NDVI as an advisor would have seen it on a chosen day."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from common import (
    BROWN_GREEN,
    STEP_DAYS,
    VIRIDIS,
    YELLOW_RED,
    colour_scale,
    fields_meta,
    legend,
    matrices,
    plot_table,
    plots,
)

COLOUR_BY = {
    "Estimated NDVI": "ndvi",
    "Difference from the same crop": "anomaly",
    "Days since the last clear image": "age",
}


def state_on(step: int) -> pd.DataFrame:
    """What was known about every field on the given step, using only images up to that day."""
    m, table = matrices(), plot_table()
    observed = ~np.isnan(m["observed"][:, : step + 1])
    last = np.where(observed.any(axis=1), step - np.argmax(observed[:, ::-1], axis=1), -1)
    estimate = m["real_time"][:, step]
    df = table[["crop", "area_ha"]].copy()
    df["ndvi"] = estimate
    df["low"], df["high"] = m["real_time_low"][:, step], m["real_time_high"][:, step]
    df["median"] = df.groupby("crop")["ndvi"].transform("median")
    df["anomaly"] = df["ndvi"] - df["median"]
    df["age"] = np.where(last >= 0, (step - last) * STEP_DAYS, np.nan)
    df["clear_today"] = last == step
    return df


def field_map(df: pd.DataFrame, colour_by: str) -> str | None:
    values = df[colour_by].to_numpy()
    if colour_by == "ndvi":
        colours = colour_scale(values, 0.1, 0.9, VIRIDIS)
    elif colour_by == "anomaly":
        colours = colour_scale(values, -0.25, 0.25, BROWN_GREEN)
    else:
        colours = colour_scale(values, 0, 60, YELLOW_RED)
    labels = {
        "ndvi": [f"NDVI {v:.2f}" if not np.isnan(v) else "no estimate yet" for v in values],
        "anomaly": [f"{v:+.2f} vs crop median" if not np.isnan(v) else "-" for v in values],
        "age": [f"last clear image {v:.0f} days ago" if not np.isnan(v) else "-" for v in values],
    }[colour_by]
    geojson = plots()
    features = []
    for feature, colour, label in zip(geojson["features"], colours, labels, strict=True):
        props = {**feature["properties"], "fill": colour, "label": label}
        features.append({**feature, "properties": props})
    bounds = np.array(
        [c for f in features for ring in f["geometry"]["coordinates"] for c in _flat(ring)]
    )
    layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": features},
        id="fields",
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill",
        get_line_color=[60, 60, 60, 120],
        line_width_min_pixels=0.5,
        auto_highlight=True,
    )
    view = pdk.ViewState(
        longitude=float(bounds[:, 0].mean()), latitude=float(bounds[:, 1].mean()), zoom=10.6
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"html": "<b>Plot {parcel_id}</b>, {crop}<br/>{label}"},
        map_style="light",
    )
    event = st.pydeck_chart(
        deck, on_select="rerun", selection_mode="single-object", key="field_map", height=480
    )
    picked = (event.selection or {}).get("objects", {}).get("fields", [])
    return picked[0]["properties"]["parcel_id"] if picked else None


def _flat(ring):
    """Coordinates of a polygon or multipolygon ring, at any nesting depth."""
    if ring and isinstance(ring[0], (int, float)):
        yield ring
    else:
        for part in ring:
            yield from _flat(part)


def field_chart(parcel_id: int, step: int, show_hindsight: bool) -> None:
    m, table, dates = matrices(), plot_table(), fields_meta()["dates"]
    row, crop = table.index.get_loc(parcel_id), table.loc[parcel_id, "crop"]
    crop_rows = np.flatnonzero(table["crop"].to_numpy() == table.loc[parcel_id, "crop"])
    band = np.nanpercentile(m["real_time"][crop_rows], [25, 50, 75], axis=0)
    observed, real_time = m["observed"][row], m["real_time"][row]
    past = np.arange(len(dates)) <= step

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=band[2], line={"width": 0}, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=band[0],
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.25)",
            line={"width": 0},
            name="Same crop: middle half of the fields",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=band[1],
            line={"color": "#888", "width": 1, "dash": "dot"},
            name="Same crop: median",
        )
    )
    if show_hindsight:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=m["after_season"][row],
                line={"color": "#0072B2", "dash": "dash"},
                name="End-of-season reconstruction (hindsight)",
            )
        )
    low, high = m["real_time_low"][row], m["real_time_high"][row]
    fig.add_trace(
        go.Scatter(
            x=dates[past], y=high[past], line={"width": 0}, showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates[past],
            y=low[past],
            fill="tonexty",
            fillcolor="rgba(213,94,0,0.18)",
            line={"width": 0},
            name="Likely range (80 %)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates[past],
            y=real_time[past],
            line={"color": "#D55E00", "width": 2.5},
            name="Real-time estimate (only past images)",
        )
    )
    seen = past & ~np.isnan(observed)
    fig.add_trace(
        go.Scatter(
            x=dates[seen],
            y=observed[seen],
            mode="markers",
            marker={"color": "black", "size": 7},
            name="Clear image",
        )
    )
    future = ~past & ~np.isnan(observed)
    fig.add_trace(
        go.Scatter(
            x=dates[future],
            y=observed[future],
            mode="markers",
            name="Later clear image (not known yet)",
            marker={"color": "white", "size": 7, "line": {"color": "#555", "width": 1}},
        )
    )
    fig.add_vline(x=dates[step], line={"color": "#333", "width": 1})
    for frost in fields_meta().get("frost_dates", []):
        fig.add_vline(x=pd.Timestamp(frost), line={"color": "#56B4E9", "width": 1, "dash": "dot"})
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="NDVI",
        yaxis_range=[0, 1],
        legend={"orientation": "h", "y": -0.15},
        hovermode="x unified",
        title=f"Plot {parcel_id}, {crop} ({table.loc[parcel_id, 'area_ha']} ha)",
    )
    st.plotly_chart(fig)


def render() -> None:
    meta, table = fields_meta(), plot_table()
    dates = meta["dates"]
    st.title("Field monitor")
    st.markdown(
        f"{len(table):,} winter wheat and rapeseed fields east of Reims, season {meta['season']}. "
        "Pick a day: the map shows what an agronomist would have known **that day**, with the "
        "fields under clouds estimated from their last clear image and from Sentinel-1 radar, "
        "which sees through clouds."
    )

    c1, c2, c3 = st.columns([2.2, 1, 1.3])
    default = int(np.searchsorted(dates, pd.Timestamp("2025-02-08")))  # cloudy for weeks, no frost
    day = c1.select_slider(
        "Day",
        options=list(dates),
        value=dates[default],
        format_func=lambda d: d.strftime("%d %b %Y"),
    )
    step = int(np.searchsorted(dates, day))
    crop = c2.radio("Crop", ["Both", "Winter wheat", "Winter rapeseed"], horizontal=False)
    colour_label = c3.radio("Colour the fields by", list(COLOUR_BY))
    colour_by = COLOUR_BY[colour_label]

    df = state_on(step)
    frost_day = day.date().isoformat() in meta.get("frost_dates", [])
    if frost_day:
        st.info(
            "Frost day: the radar of the whole area dropped at once, as frozen plants reflect "
            "radar differently. Today's estimates ignore the radar and rely on the last clear "
            "image."
        )
    shown = df if crop == "Both" else df[df["crop"] == crop]

    k1, k2, k3 = st.columns(3)
    k1.metric("Fields with a clear image that day", f"{shown['clear_today'].mean():.0%}")
    k2.metric("Last clear image older than 3 weeks", f"{(shown['age'] > 21).mean():.0%}")
    k3.metric("Median estimated NDVI", f"{shown['ndvi'].median():.2f}")

    left, right = st.columns([1.6, 1])
    with left:
        mask = df.index.isin(shown.index)
        map_df = df.copy()
        map_df.loc[~mask, colour_by] = np.nan
        picked = field_map(map_df, colour_by)
        if colour_by == "ndvi":
            legend(
                [
                    ("0.1 bare soil", VIRIDIS[0]),
                    ("0.5", VIRIDIS[2]),
                    ("0.9 dense crop", VIRIDIS[4]),
                    ("no estimate yet", "#c8c8c8"),
                ]
            )
        elif colour_by == "anomaly":
            legend(
                [
                    ("-0.25 behind its crop", BROWN_GREEN[0]),
                    ("same as the median", BROWN_GREEN[2]),
                    ("+0.25 ahead", BROWN_GREEN[4]),
                ]
            )
        else:
            legend(
                [("today", YELLOW_RED[0]), ("30 days", YELLOW_RED[2]), ("60+ days", YELLOW_RED[4])]
            )
        st.caption("Click a field to see its season below.")

    with right:
        st.subheader("Fields to check")
        st.caption(
            "The fields furthest below the median of their crop that day. Age: days since the "
            "last clear image. Click a row to see the field."
        )
        check = shown.dropna(subset=["anomaly"]).nsmallest(10, "anomaly")
        if frost_day:
            st.warning("Paused on frost days: the estimates are less reliable for ranking fields.")
            check = check.iloc[:0]
        check_view = pd.DataFrame(
            {
                "Plot": check.index,
                "Crop": check["crop"].str.replace("Winter ", ""),
                "NDVI": check["ndvi"].round(2),
                "vs crop": check["anomaly"].round(2),
                "Age (d)": check["age"].astype("Int64"),
            }
        )
        selection = st.dataframe(
            check_view,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            width="stretch",
            key="check_table",
        )
        rows = selection.selection.get("rows", [])
        if rows:
            picked = int(check_view.iloc[rows[0]]["Plot"])

    if picked is not None:
        st.session_state["field"] = int(picked)
    field = st.session_state.get("field", int(check.index[0]) if len(check) else table.index[0])
    field = st.selectbox(
        "Field",
        table.index,
        index=int(table.index.get_loc(field)),
        format_func=lambda p: f"Plot {p} ({table.loc[p, 'crop']})",
    )
    show_hindsight = st.toggle("Show the end-of-season reconstruction (hindsight)", value=False)
    field_chart(int(field), step, show_hindsight)

    info = df.loc[field]
    age = "no clear image yet" if np.isnan(info["age"]) else f"{info['age']:.0f} days old"
    likely = ""
    if not info["clear_today"] and not np.isnan(info["low"]):
        likely = f" (likely between {info['low']:.2f} and {info['high']:.2f})"
    st.markdown(
        f"On **{day:%d %B %Y}**, the last clear image of this field was **{age}**. "
        f"Estimated NDVI: **{info['ndvi']:.2f}**{likely}, against **{info['median']:.2f}** for "
        f"the median {info['crop'].lower()} field."
    )

    d1, d2 = st.columns(2)
    m = matrices()
    row = table.index.get_loc(field)
    field_csv = pd.DataFrame(
        {
            "date": dates.date,
            "clear_image_ndvi": m["observed"][row],
            "real_time_estimate": m["real_time"][row],
            "end_of_season_reconstruction": m["after_season"][row],
        }
    ).to_csv(index=False)
    d1.download_button("Download this field's season (CSV)", field_csv, f"plot_{field}.csv")
    day_csv = df.reset_index().rename(columns={"index": "parcel_id"}).to_csv(index=False)
    d2.download_button(
        f"Download all fields on {day:%d %b %Y} (CSV)", day_csv, f"fields_{day:%Y-%m-%d}.csv"
    )

    trained_on = ", ".join(meta["models_trained_on"])
    with st.expander("How the estimates are made, and their limits"):
        st.markdown(
            f"""
- **Real-time estimate** (orange): on each day without a clear image, a model predicts NDVI from the
  field's last clear value, the days since, and today's Sentinel-1 radar. It never uses later
  images, so it is what a service could have shown that day.
- **End-of-season reconstruction** (blue, dashed): once the season is over, a second model also
  uses the next clear image after each gap. It is more accurate, but only known in hindsight.
- **Models trained on {trained_on} only**, then applied to this season, as a real service would
  be. Typical error under real clouds: about 0.08 NDVI, larger after long gaps.
  Enough to tell a field's stage, not subtle stress. Best from April to the harvest; weakest for
  wheat greening up in late winter after weeks of clouds, where the estimate tends to run high
  (see "Try the gap-filling").
- **Does the list of fields to check work?** On 8 February 2025, after three weeks of clouds, it
  flagged 10 fields. Ten days later, a clear image showed 9 of them really behind their crop (6
  among the lowest 4 % of 1,247 fields) and 1 false alarm.
- **Frost days** (dotted blue lines on the chart): when the radar of the whole area drops at once in
  winter, the real-time estimate ignores it. Frozen plants fool the radar: on the one frost day with
  clear images to check, radar tripled the real-time error.
- **Coverage:** winter wheat and winter rapeseed, east of Reims. Other crops and regions were not
  tested.
- A field with fewer than {meta["min_radar_passes"]} radar passes is estimated without radar.
"""
        )
