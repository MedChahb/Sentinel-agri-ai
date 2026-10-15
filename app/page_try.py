"""Try the gap-filling: hide a period of a real field, let the live API rebuild it."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import ApiError, api_url, call_api, clear_ndvi, plot_table, plots, radar

DEFAULT_FIELD = 469  # a wheat field whose May-June result is the median of a sample of fields


def build_request(field: int, start, end, mode: str, use_radar: bool):
    obs = clear_ndvi()
    obs = obs[obs["parcel_id"] == field].sort_values("date")
    hidden = obs["date"].between(start, end)
    visible = obs[~hidden]
    request = {
        "crop": plot_table().loc[field, "crop"],
        "mode": mode,
        "ndvi": [
            {"date": d.date().isoformat(), "ndvi": round(float(v), 4)}
            for d, v in zip(visible["date"], visible["ndvi"], strict=True)
        ],
        "radar": [],
    }
    if use_radar:
        r = radar()
        r = r[r["parcel_id"] == field]
        request["radar"] = [
            {
                "date": d.date().isoformat(),
                "vv_db": round(float(vv), 2),
                "vh_db": round(float(vh), 2),
                "orbit": int(o),
            }
            for d, vv, vh, o in zip(r["date"], r["vv_db"], r["vh_db"], r["orbit"], strict=True)
        ]
    return request, visible, obs[hidden]


def _rings(geometry: dict):
    """Outer rings of a Polygon or MultiPolygon, as (lons, lats)."""
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]
    for polygon in polygons:
        ring = np.array(polygon[0])
        yield ring[:, 0], ring[:, 1]


def field_location(field: int) -> None:
    """The chosen field outlined on a satellite view, with its neighbours."""
    lons, lats, own = [], [], None
    for feature in plots()["features"]:
        for x, y in _rings(feature["geometry"]):
            if feature["properties"]["parcel_id"] == field:
                own = (x, y)
            else:
                lons += [*x, None]
                lats += [*y, None]
    x, y = own
    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode="lines",
            hoverinfo="skip",
            line={"color": "rgba(255,255,255,0.6)", "width": 1},
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=x,
            lat=y,
            mode="lines",
            fill="toself",
            hoverinfo="skip",
            fillcolor="rgba(230,159,0,0.25)",
            line={"color": "#E69F00", "width": 3},
        )
    )
    span = max(np.ptp(x), np.ptp(y) * 1.5)
    zoom = float(np.clip(np.log2(360 / max(span, 1e-4)) - 1.6, 11, 17))
    fig.update_layout(
        map={
            "style": "satellite",
            "center": {"lon": float(x.mean()), "lat": float(y.mean())},
            "zoom": zoom,
        },
        height=260,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("The field (orange) on a satellite view, with the neighbouring fields of the study.")


def render() -> None:
    table = plot_table()
    st.title("Try the gap-filling")
    st.markdown(
        "Clouds hide a field for weeks. Here **you** play the clouds: choose a real field and a "
        "period to hide. The app sends the remaining clear images (and, if you want, the radar) "
        "to the model API on AWS, which rebuilds the curve. The hidden images are the truth it is "
        "scored against."
    )

    c1, c2 = st.columns([1, 1.6])
    field = c1.selectbox(
        "Field",
        table.index,
        index=int(table.index.get_loc(DEFAULT_FIELD)),
        format_func=lambda p: f"Plot {p} ({table.loc[p, 'crop']})",
    )
    n_clear = int((clear_ndvi()["parcel_id"] == field).sum())
    c1.markdown(
        f"**{table.loc[field, 'crop']}**, {table.loc[field, 'area_ha']} ha, east of Reims.  \n"
        f"Season 2024–25: **{n_clear}** clear Sentinel-2 images and "
        f"**{table.loc[field, 'radar_passes']}** Sentinel-1 radar passes."
    )
    with c2:
        field_location(int(field))
    dates = pd.date_range("2024-10-01", "2025-08-31", freq="D")
    start, end = st.select_slider(
        "Period hidden by clouds",
        options=list(dates),
        value=(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30")),
        format_func=lambda d: d.strftime("%d %b %Y"),
    )
    c3, c4 = st.columns(2)
    mode_label = c3.radio(
        "Mode",
        ["After the season (images before and after the gap)", "Real time (only images before)"],
    )
    mode = "after_season" if mode_label.startswith("After") else "real_time"
    use_radar = c4.toggle("Use Sentinel-1 radar", value=True)

    request, visible, hidden = build_request(int(field), start, end, mode, use_radar)
    if mode == "real_time":  # in real time, nothing after the gap is known yet
        visible = visible[visible["date"] < start]
        request["ndvi"] = [o for o in request["ndvi"] if o["date"] < start.date().isoformat()]
    st.caption(
        f"{len(visible)} clear images sent, {len(hidden)} hidden, "
        f"{len(request['radar'])} radar passes sent."
    )

    if len(request["ndvi"]) < 2:
        st.warning("Keep at least two clear images visible.")
        return
    if not len(hidden):
        st.info("No clear image falls in this period: widen it to have something to score.")

    # The model is called again whenever a setting changes; the last answer is kept meanwhile.
    answer, sent = st.session_state.get("try_answer", (None, None))
    if answer is None or sent != request:
        with st.spinner(
            "Calling the model on AWS. After a quiet period, it takes about 10 s to wake up."
        ):
            try:
                answer = call_api("/gapfill", json=request)
            except ApiError as err:
                st.error(str(err))
                return
        st.session_state["try_answer"] = (answer, request)

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(answer["dates"]),
            "ndvi": answer["ndvi"],
            "observed": answer["observed"],
        }
    ).astype({"ndvi": float})
    step_of = lambda d: (d - result["date"][0]).days // 5  # noqa: E731
    truth = hidden.assign(step=hidden["date"].map(step_of))
    truth = truth[truth["step"] < len(result)]
    model_values = result["ndvi"].to_numpy()[truth["step"].to_numpy()]
    x_visible = visible["date"].map(pd.Timestamp.toordinal).to_numpy()
    line = np.interp(truth["date"].map(pd.Timestamp.toordinal), x_visible, visible["ndvi"])
    if mode == "real_time":  # the real-time baseline: keep the last clear value
        line = np.full(len(truth), visible["ndvi"].iloc[-1])

    fig = go.Figure()
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor="rgba(150,150,150,0.2)",
        line_width=0,
        annotation_text="hidden",
        annotation_position="top left",
    )
    baseline_name = "Straight line" if mode == "after_season" else "Keep the last clear value"
    if mode == "after_season":
        fig.add_trace(
            go.Scatter(
                x=visible["date"],
                y=visible["ndvi"],
                name=baseline_name,
                line={"color": "#999", "dash": "dash"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=result["date"],
            y=result["ndvi"],
            name="Model (API)",
            line={"color": "#D55E00", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=visible["date"],
            y=visible["ndvi"],
            mode="markers",
            marker={"color": "black", "size": 7},
            name="Clear image sent",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=truth["date"],
            y=truth["ndvi"],
            mode="markers",
            name="Hidden clear image (truth)",
            marker={"color": "white", "size": 9, "line": {"color": "#CC0000", "width": 2}},
        )
    )
    fig.update_layout(
        height=420,
        yaxis_title="NDVI",
        yaxis_range=[0, 1],
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig)

    if len(truth) and not np.isnan(model_values).all():
        ok = ~np.isnan(model_values)
        model_mae = np.abs(model_values[ok] - truth["ndvi"].to_numpy()[ok]).mean()
        line_mae = np.abs(line[ok] - truth["ndvi"].to_numpy()[ok]).mean()
        k1, k2, k3 = st.columns(3)
        k1.metric("Model error (mean, NDVI)", f"{model_mae:.3f}")
        k2.metric(f"{baseline_name}: error", f"{line_mae:.3f}")
        k3.metric("Radar used", "yes" if answer["radar_used"] else "no")
        st.caption(
            "Errors on the hidden clear images, for this one field. Over all fields, the model "
            "is 17 % better than the straight line under real clouds, but not everywhere: see "
            "below."
        )

    with st.expander("Where the model helps, and where it fails"):
        st.markdown(
            """
Measured over every field of 2024-25, hiding two months at a time (after-season model, with radar):

| Hidden period | Wheat: model vs straight line | Rapeseed |
|---|---|---|
| Nov-Dec | about equal (-4 %) | +18 % |
| Jan-Feb | -16 % | -32 % |
| mid-Feb to mid-Apr | **-71 %** | +11 % |
| Apr-May | **+61 %** | +31 % |
| mid-May to mid-Jul | +27 % | **+63 %** |
| mid-Jun to mid-Aug | +38 % | +48 % |

- **It helps most from April to the harvest**, when the curve bends (peak, flowering, ripening,
  harvest) and a straight line cuts the corner.
- **Known weak spot: wheat greening up in late winter after a long gap.** The models were trained on
  2023-24, when wheat greened up a month earlier, and they still overshoot by about 0.13. Try
  15 February to 15 April on a wheat field to see it. The fix is a clock that follows the weather
  (growing degree days) instead of the calendar, and more training seasons.
"""
        )

    with st.expander("What the app sent to the API"):
        st.markdown(f"`POST {api_url()}/gapfill`. The same request from a terminal:")
        st.code(
            f"curl -X POST {api_url()}/gapfill -H 'Content-Type: application/json' "
            "-d @request.json",
            language="bash",
        )
        preview = {
            **request,
            "ndvi": request["ndvi"][:3] + ["..."],
            "radar": request["radar"][:2] + ["..."],
        }
        st.code(json.dumps(preview, indent=1), language="json")
