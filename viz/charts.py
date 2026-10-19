"""Altair charts for the dashboard (rendered with Streamlit's theme, so axes, fonts and
gridlines follow light / dark mode). Colours come from the validated categorical
palette: slot 1 blue for amounts, slot 2 orange for deductions, a neutral grey
track for "requested", and a single-hue blue ramp for counts."""
from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = {"light": {"blue": "#2a78d6", "orange": "#eb6834", "track": "#c3c2b7", "ink": "#101828", "muted": "#52514e"},
           "dark": {"blue": "#3987e5", "orange": "#d95926", "track": "#4a4a46", "ink": "#e7eaf0", "muted": "#c3c2b7"}}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ₹ axis ticks in Indian units: 50K, 1.5L, 10L
RUPEE_TICKS = ("datum.value >= 100000 ? '₹' + format(datum.value / 100000, '.3~f') + 'L' : "
               "datum.value >= 1000 ? '₹' + format(datum.value / 1000, '.3~f') + 'K' : '₹' + datum.value")


def _p(mode: str) -> dict:
    return PALETTE["dark" if mode == "dark" else "light"]


def waterfall(steps: list[dict], inr, mode: str = "light") -> alt.Chart:
    """Requested estimate → each deduction → payable, as a horizontal waterfall."""
    p, rows, running = _p(mode), [], 0
    for i, s in enumerate(steps):
        first, last = i == 0, i == len(steps) - 1
        if first or last:
            lo, hi, kind = 0, s["amount"], "Amount"
            running = s["amount"]
        else:
            lo, hi, kind = running + s["amount"], running, "Deduction"
            running += s["amount"]
        rows.append({"step": f"{i + 1}. {s['step']}", "lo": lo, "hi": hi, "kind": kind,
                     "label": inr(s["amount"]) if kind == "Amount" else "−" + inr(-s["amount"]),
                     "note": s.get("note") or ""})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("step:N", sort=None, title=None, axis=alt.Axis(labelLimit=280, ticks=False, domain=False)))
    bars = base.mark_bar(size=20, cornerRadius=4).encode(
        x=alt.X("lo:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=5, grid=True),
                scale=alt.Scale(domain=[0, df["hi"].max() * 1.22], nice=False)), x2="hi:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Amount", "Deduction"], range=[p["blue"], p["orange"]]),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        tooltip=[alt.Tooltip("step:N", title="Step"), alt.Tooltip("label:N", title="Amount"), alt.Tooltip("note:N", title="Why")])
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color=p["ink"]).encode(x="hi:Q", text="label:N")
    return (bars + labels).properties(height=80 + 46 * len(df))  # rows + legend and axis


def authorised_vs_requested(df: pd.DataFrame, inr, mode: str = "light") -> alt.Chart:
    """Bullet bars: grey track = requested, blue bar = authorised, one row per paid claim."""
    p = _p(mode)
    long = pd.concat([pd.DataFrame({"claim": df["Request"], "series": "Requested", "value": df["Requested"]}),
                      pd.DataFrame({"claim": df["Request"], "series": "Authorised", "value": df["Authorised"]})])
    long["label"] = long["value"].map(inr)
    color = alt.Color("series:N", scale=alt.Scale(domain=["Requested", "Authorised"], range=[p["track"], p["blue"]]),
                      legend=alt.Legend(orient="top", title=None, symbolType="square"))
    base = alt.Chart(long).encode(y=alt.Y("claim:N", title=None, sort=None, axis=alt.Axis(ticks=False, domain=False)),
                                  tooltip=[alt.Tooltip("claim:N", title="Request"), alt.Tooltip("series:N", title="Series"),
                                           alt.Tooltip("label:N", title="Amount")])
    x = alt.X("value:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=5),
              scale=alt.Scale(domain=[0, max(1, long["value"].max()) * 1.5], nice=False))  # room for end labels
    track = base.transform_filter("datum.series == 'Requested'").mark_bar(size=20, cornerRadius=4).encode(x=x, color=color)
    paid = base.transform_filter("datum.series == 'Authorised'").mark_bar(size=10, cornerRadius=3).encode(x=x, color=color)
    req = df.assign(label=df["Authorised"].map(inr))
    labels = alt.Chart(req).mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color=p["ink"]).encode(
        y=alt.Y("Request:N", sort=None), x="Requested:Q", text="label:N")
    return (track + paid + labels).properties(height=80 + 40 * len(df))


def confusion_heatmap(cm: pd.DataFrame) -> alt.Chart:
    """Expected decision (rows) vs agents' decision (columns); the diagonal is correct."""
    order = list(cm.index)
    long = cm.reset_index(names="Expected").melt(id_vars="Expected", var_name="Agents", value_name="n")
    top = max(1, int(long["n"].max()))
    base = alt.Chart(long).encode(
        x=alt.X("Agents:N", sort=order, title="Agents decided",
                axis=alt.Axis(labelAngle=-35, labelAlign="right", ticks=False, domain=False, labelOverlap=False, labelFontSize=10)),
        y=alt.Y("Expected:N", sort=order, title="Expected", axis=alt.Axis(ticks=False, domain=False, labelFontSize=10)),
        tooltip=[alt.Tooltip("Expected:N"), alt.Tooltip("Agents:N", title="Agents decided"), alt.Tooltip("n:Q", title="Cases")])
    cells = base.mark_rect(cornerRadius=6, stroke="#ffffff00").encode(
        color=alt.condition("datum.n > 0", alt.Color("n:Q", scale=alt.Scale(domain=[0, top], range=BLUE_RAMP[1:6]), legend=None),
                            alt.value("rgba(127,127,127,0.10)")))
    text = base.mark_text(fontSize=15, fontWeight=700).encode(
        text=alt.condition("datum.n > 0", alt.Text("n:Q"), alt.value("")),
        color=alt.condition(f"datum.n >= {top / 2}", alt.value("#ffffff"), alt.value("#0d366b")))
    return (cells + text).properties(height=280, width=alt.Step(58))  # angled labels never collide


def tariff_ranges(tariffs: dict, inr, mode: str = "light") -> alt.Chart:
    """Network price range (min to max) per package procedure."""
    p = _p(mode)
    rows = [{"procedure": v["label"], "min": v["min"], "max": v["max"], "range": f"{inr(v['min'])} – {inr(v['max'])}"}
            for v in tariffs.values() if v["basis"] == "package"]
    df = pd.DataFrame(rows).sort_values("max")
    base = alt.Chart(df).encode(y=alt.Y("procedure:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
                                tooltip=[alt.Tooltip("procedure:N", title="Procedure"), alt.Tooltip("range:N", title="Network tariff")])
    bars = base.mark_bar(size=14, cornerRadius=7, color=p["blue"]).encode(
        x=alt.X("min:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=6),
                scale=alt.Scale(domain=[0, df["max"].max() * 1.42], nice=False)), x2="max:Q")
    labels = base.mark_text(align="left", dx=8, fontSize=11.5, color=p["muted"]).encode(x="max:Q", text="range:N")
    return (bars + labels).properties(height=50 + 34 * len(df))
