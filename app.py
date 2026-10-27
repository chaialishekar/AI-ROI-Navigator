"""AI Use-Case ROI Navigator

An independent concept project: helps a company decide which AI use case to deploy,
prove its value in dollars, and check whether it is ready for production.

Tab 1  Prioritize   - score use cases on value vs feasibility
Tab 2  Prove value  - tune a real predictive maintenance model for dollars, not accuracy
Tab 3  Readiness    - pilot-to-production readiness check
Tab 4  Business case - one-page summary to share with finance
Tab 5  Track value  - after go-live: forecast vs actual savings, adoption and drift
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring import (
    DEFAULT_FEAS_WEIGHTS,
    DEFAULT_VALUE_WEIGHTS,
    FEASIBILITY_CRITERIA,
    THRESHOLD,
    VALUE_CRITERIA,
    score,
)
from src.model import train_and_score
from src.readiness import ACTIONS, DIMENSIONS, verdict
from src.tracking import ADOPTION_FLOOR, COLUMNS, DRIFT_DROP, DRIFT_WINDOW, explain_gap, track, validate
from src.value import (
    ASSUMPTIONS,
    Costs,
    bootstrap,
    default_ranges,
    evaluate,
    plain_language,
    scenarios,
    sensitivity,
    sweep,
)

st.set_page_config(page_title="AI Use-Case ROI Navigator", layout="wide")

QUADRANT_COLORS = {
    "Quick win": "#1d7a45",
    "Big bet": "#0f6d74",
    "Fill-in": "#a25f00",
    "Deprioritize": "#8a979c",
}


@st.cache_data
def load_use_cases() -> pd.DataFrame:
    return pd.read_csv("data/use_cases.csv")


st.title("AI Use-Case ROI Navigator")
st.caption(
    "Most AI projects stall between pilot and production. This tool helps pick the right use case, "
    "prove its value in dollars, and check readiness to go live. Independent concept project; "
    "scores are analyst assumptions you can edit."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 · Prioritize", "2 · Prove value", "3 · Readiness", "4 · Business case", "5 · Track value"]
)


@st.cache_resource
def get_model_results():
    return train_and_score()


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x/1e6:.2f}M"
    return f"{sign}${x/1e3:,.0f}k" if x >= 1e3 else f"{sign}${x:,.0f}"


def md(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't render dollar amounts as math."""
    return text.replace("$", "\\$")


def months(x: float) -> str:
    return f"{x:.1f} months" if np.isfinite(x) else "Never"


def money_range(lo_hi: tuple) -> str:
    return f"{money(lo_hi[0])} – {money(lo_hi[1])}"


# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Which AI use case should we do first?")
    st.write(
        "Each use case is scored 1–5 on **value** and **feasibility**. "
        "High on both = **quick win**. Edit any score or weight to test your own assumptions."
    )

    with st.sidebar:
        st.header("Prioritize weights")
        st.caption("How much each criterion counts. Weights in each group are normalized.")
        st.markdown("**Value**")
        value_w = {k: st.slider(lbl, 0.0, 1.0, DEFAULT_VALUE_WEIGHTS[k], 0.05, key=f"vw_{k}")
                   for k, lbl in VALUE_CRITERIA.items()}
        st.markdown("**Feasibility**")
        feas_w = {k: st.slider(lbl, 0.0, 1.0, DEFAULT_FEAS_WEIGHTS[k], 0.05, key=f"fw_{k}")
                  for k, lbl in FEASIBILITY_CRITERIA.items()}

    industries = sorted(load_use_cases()["industry"].unique())
    picked = st.multiselect("Industries", industries, default=industries)

    base = load_use_cases()
    base = base[base["industry"].isin(picked)]

    with st.expander("Edit scores (1–5, 5 is always better)"):
        edited = st.data_editor(
            base,
            hide_index=True,
            width="stretch",
            disabled=["use_case", "industry", "description"],
            column_config={
                c: st.column_config.NumberColumn(lbl, min_value=1, max_value=5, step=1)
                for c, lbl in {**VALUE_CRITERIA, **FEASIBILITY_CRITERIA}.items()
            },
        )

    if edited.empty:
        st.info("Pick at least one industry.")
        scored = score(load_use_cases(), value_w, feas_w)
    else:
        scored = score(edited, value_w, feas_w)
    scored.insert(0, "rank", range(1, len(scored) + 1))

    # Nudge tied points apart so every marker stays visible (display only)
    plot_df = scored.copy()
    dup = plot_df.groupby(["feasibility", "value"]).cumcount()
    plot_df["x_plot"] = plot_df["feasibility"] + dup * 0.07
    plot_df["y_plot"] = plot_df["value"] - dup * 0.07

    st.session_state["top_use_case"] = scored.iloc[0]["use_case"]
    st.session_state["top_quadrant"] = scored.iloc[0]["quadrant"]

    # Headline: top 3 quick wins
    wins = scored[scored["quadrant"] == "Quick win"].head(3)
    cols = st.columns(3)
    for col, (_, r) in zip(cols, wins.iterrows()):
        col.metric(r["use_case"], f"{r['priority']:.1f}", help=f"Value {r['value']} × Feasibility {r['feasibility']}")
        col.caption(f"{r['industry']} · quick win")

    fig = px.scatter(
        plot_df,
        x="x_plot",
        y="y_plot",
        color="quadrant",
        color_discrete_map=QUADRANT_COLORS,
        text="rank",
        hover_name="use_case",
        hover_data={"x_plot": False, "y_plot": False, "rank": False, "industry": True,
                    "value": True, "feasibility": True, "priority": True},
        range_x=[1, 5.2],
        range_y=[1, 5.2],
        labels={"x_plot": "Feasibility →", "y_plot": "Value →"},
        height=560,
    )
    fig.add_vline(x=THRESHOLD, line_dash="dot", line_color="#8a979c")
    fig.add_hline(y=THRESHOLD, line_dash="dot", line_color="#8a979c")
    for x, y, label in [(4.6, 5.1, "QUICK WINS"), (2.2, 5.1, "BIG BETS"), (4.6, 1.15, "FILL-INS"), (2.2, 1.15, "DEPRIORITIZE")]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=11, color="#8a979c"))
    fig.update_traces(textposition="middle center", marker=dict(size=24), textfont=dict(color="white", size=11))
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.caption("Numbers on the chart match the rank in the list below. Hover a point for details.")
    st.markdown("**Ranked list**")
    st.dataframe(
        scored[["rank", "use_case", "industry", "value", "feasibility", "priority", "quadrant", "next_step"]],
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("What is predictive maintenance worth to this plant?")
    st.write(
        "A model trained on 10,000 real-style machine records predicts which machines are about to fail. "
        "Enter your plant's costs, then move the **alert threshold** to see how dollars change. "
        "The most accurate setting is not always the most profitable one."
    )

    res = get_model_results()
    y_test = res["y_test"]

    with st.expander("Your plant's costs (edit to match your operation)", expanded=True):
        c1, c2, c3 = st.columns(3)
        costs = Costs(
            failures_per_year=c1.number_input("Unplanned failures per year today", 1, 10_000, 40, 1, key="c_fail"),
            cost_missed=c1.number_input("Cost of an unplanned failure ($)", 1, 10_000_000, 25_000, 1_000, key="c_miss"),
            cost_caught=c2.number_input("Cost of a planned repair when caught early ($)", 1, 10_000_000, 4_000, 500, key="c_caught"),
            cost_false_alarm=c2.number_input("Cost of a false alarm inspection ($)", 1, 1_000_000, 500, 50, key="c_fa"),
            running_cost=c3.number_input("Yearly running cost of the AI ($)", 1, 10_000_000, 60_000, 5_000, key="c_run"),
            deployment_cost=c3.number_input("One-time deployment cost ($)", 1, 10_000_000, 150_000, 10_000, key="c_dep"),
            months_to_deploy=c3.number_input("Months until live", 0, 24, 2, 1, key="c_months"),
        )
        st.caption("Defaults are illustrative assumptions for a mid-size plant. All values must be greater than 0.")

    model_name = st.radio("Model", list(res["models"].keys()), index=1, horizontal=True, key="model_pick")
    proba = res["models"][model_name]["proba"]
    curve = sweep(y_test, proba, costs)
    best_money = curve.loc[curve["annual_net"].idxmax()]
    best_acc = curve.loc[curve["accuracy"].idxmax()]

    threshold = st.slider(
        "Alert threshold: alert when failure probability is at least…",
        0.01, 0.99, float(round(best_money["threshold"], 2)), 0.01, key="thr",
        help="Lower = more sensitive: catches more failures but raises more false alarms.",
    )
    r = evaluate(y_test, proba, threshold, costs)
    band = bootstrap(y_test, proba, threshold, costs)
    band_help = ("90% range: where the result would likely land if the model were tested on a "
                 "different sample of machines (bootstrap of the test set). Your cost inputs are held fixed.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annual net savings", money(r["annual_net"]), help=band_help)
    m1.caption(md(f"90% range: {money_range(band['annual_net'])}"))
    m2.metric("Payback", months(r["payback_months"]), help=band_help)
    m2.caption(f"90% range: {months(band['payback_months'][0])} – {months(band['payback_months'][1])}")
    m3.metric("3-year ROI", f"{r['roi_3y']*100:,.0f}%", help=band_help)
    m3.caption(f"90% range: {band['roi_3y'][0]*100:,.0f}% – {band['roi_3y'][1]*100:,.0f}%")
    m4.metric("First-year value", money(r["first_year"]), help="Savings while live in year one, minus deployment cost")
    st.info(f"**In plain language:** {plain_language(r)}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Failures caught / year", f"{r['caught']:.0f}")
    k1.caption(f"90% range: {band['caught'][0]:.0f} – {band['caught'][1]:.0f}")
    k2.metric("Failures missed / year", f"{r['missed']:.0f}")
    k2.caption(f"90% range: {band['missed'][0]:.0f} – {band['missed'][1]:.0f}")
    k3.metric("False alarms / year", f"{r['false_alarms']:.0f}")
    k3.caption(f"90% range: {band['false_alarms'][0]:.0f} – {band['false_alarms'][1]:.0f}")
    k4.metric("Accuracy", f"{r['accuracy']*100:.1f}%", help="Share of all test readings classified correctly")
    st.caption(
        f"Ranges reflect **model uncertainty**: the test set has only {int(y_test.sum())} real failures, "
        "so catch and false-alarm rates are estimates. For uncertainty in your cost inputs, see "
        "*How sure are we?* below."
    )

    gap = best_money["annual_net"] - best_acc["annual_net"]
    if gap > 1:
        st.success(md(
            f"**Accuracy is not value.** The most *accurate* setting (threshold {best_acc['threshold']:.2f}, "
            f"accuracy {best_acc['accuracy']*100:.1f}%) saves **{money(best_acc['annual_net'])}** a year. "
            f"The most *profitable* setting (threshold {best_money['threshold']:.2f}, accuracy "
            f"{best_money['accuracy']*100:.1f}%) saves **{money(best_money['annual_net'])}**. "
            f"That's **{money(gap)} more a year**, because a missed failure costs far more than a false alarm."
        ))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["annual_net"], name="Annual net savings ($)",
                              line=dict(color="#0f6d74", width=3)))
    fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["accuracy"] * 100, name="Accuracy (%)",
                              line=dict(color="#a25f00", dash="dot"), yaxis="y2"))
    fig2.add_vline(x=best_money["threshold"], line_color="#1d7a45", line_dash="dash",
                   annotation_text="most profitable", annotation_position="top left")
    fig2.add_vline(x=best_acc["threshold"], line_color="#a25f00", line_dash="dash",
                   annotation_text="most accurate", annotation_position="top right")
    fig2.add_vline(x=threshold, line_color="#15222a", line_width=1, annotation_text="you", annotation_position="bottom right")
    fig2.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        xaxis_title="Alert threshold", yaxis=dict(title="Annual net savings ($)", tickformat="$,.0f"),
        yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Model comparison** (each at its most profitable threshold)")
    rows = []
    for name, m in res["models"].items():
        cv = sweep(y_test, m["proba"], costs)
        b = cv.loc[cv["annual_net"].idxmax()]
        rows.append({"Model": name, "AUC": round(m["auc"], 3), "Best threshold": round(b["threshold"], 2),
                     "Recall": f"{b['recall']*100:.0f}%", "Precision": f"{b['precision']*100:.0f}%",
                     "Annual net savings": money(b["annual_net"]), "Payback (months)": round(b["payback_months"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Custom build vs ready-made solution"):
        cc1, cc2 = st.columns(2)
        custom_m = cc1.number_input("Months to deploy: custom build", 0, 24, 6, 1, key="custom_m")
        ready_m = cc2.number_input("Months to deploy: ready-made solution", 0, 24, 1, 1, key="ready_m")
        fy = lambda months: r["annual_net"] * max(0, 12 - months) / 12 - costs.deployment_cost
        comp = pd.DataFrame({"Option": ["Custom build", "Ready-made"], "First-year value": [fy(custom_m), fy(ready_m)]})
        figc = px.bar(comp, x="Option", y="First-year value", text=comp["First-year value"].map(money),
                      color="Option", color_discrete_sequence=["#8a979c", "#0f6d74"], height=300)
        figc.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat="$,.0f")
        st.plotly_chart(figc, width="stretch")
        st.caption(md(f"Going live {custom_m - ready_m} months sooner is worth {money(fy(ready_m) - fy(custom_m))} in year one."))

    # ---------------------------------------------------------------- Sensitivity
    st.markdown("#### How sure are we? Low / base / high cases")
    st.write(
        "Your cost inputs are estimates. Set a realistic low and high value for each one "
        "to see the range of outcomes and which assumption matters most."
    )

    def fmt_input(key: str, v: float) -> str:
        return money(v) if ASSUMPTIONS[key].endswith("($)") else f"{v:,.0f}"

    base_vals = {k: float(getattr(costs, k)) for k in ASSUMPTIONS}
    defaults = default_ranges(costs)
    range_df = pd.DataFrame(
        {"Low": [defaults[k][0] for k in ASSUMPTIONS], "Base": list(base_vals.values()),
         "High": [defaults[k][1] for k in ASSUMPTIONS]},
        index=list(ASSUMPTIONS.values()),
    )
    with st.expander("Edit low and high values", expanded=False):
        edited_ranges = st.data_editor(
            range_df,
            width="stretch",
            disabled=["Base"],
            # Reset to fresh defaults whenever the base inputs above change
            key=f"sens_{hash(tuple(base_vals.values()))}",
            column_config={c: st.column_config.NumberColumn(c, min_value=0, format="%.0f") for c in range_df.columns},
        )
        st.caption("Defaults: ±25–50% around each base value, with a wider upside on deployment cost, "
                   "running cost and timeline, which are usually underestimated.")
    ranges = {k: (float(edited_ranges.loc[lbl, "Low"]), float(edited_ranges.loc[lbl, "High"]))
              for k, lbl in ASSUMPTIONS.items()}

    cases = scenarios(y_test, proba, threshold, costs, ranges)
    tornado = sensitivity(y_test, proba, threshold, costs, ranges)
    top = tornado.iloc[0]

    case_table = pd.DataFrame({
        name: {
            "Annual net savings": money(c["annual_net"]),
            "3-year net value": money(c["three_year"]),
            "3-year ROI": f"{c['roi_3y']*100:,.0f}%",
            "Payback": months(c["payback_months"]),
            "First-year value": money(c["first_year"]),
        }
        for name, c in cases.items()
    })
    st.dataframe(case_table, width="stretch")
    st.caption("Low case = every assumption at the end of its range that hurts value; "
               "high case = every assumption at the end that helps. Alert threshold held at your setting.")

    low = cases["Low"]
    if low["annual_net"] > 0 and low["payback_months"] <= 36:
        st.success(md(
            f"**Robust:** even in the low case the project saves {money(low['annual_net'])} a year "
            f"and pays back in {months(low['payback_months'])}."
        ))
    else:
        st.warning(md(
            f"**At risk:** in the low case the project {'loses' if low['annual_net'] <= 0 else 'saves only'} "
            f"{money(abs(low['annual_net']))} a year and "
            f"{'never pays back' if not np.isfinite(low['payback_months']) else 'takes ' + months(low['payback_months']) + ' to pay back'}. "
            "Firm up the top assumptions below before committing."
        ))

    labels = [f"{a.replace(' ($)', '')}  ({fmt_input(k, lo)} → {fmt_input(k, hi)})"
              for a, k, lo, hi in zip(tornado["assumption"], tornado["key"], tornado["low_input"], tornado["high_input"])]
    figt = go.Figure()
    for side, name, color in [("low", "Input at low value", "#b4322c"), ("high", "Input at high value", "#0f6d74")]:
        figt.add_trace(go.Bar(
            y=labels, x=tornado[f"at_{side}"] - tornado["base"], base=tornado["base"],
            orientation="h", name=name, marker_color=color,
            customdata=tornado[f"at_{side}"],
            hovertemplate="%{y}<br>3-year net value: %{customdata:$,.0f}<extra>" + name + "</extra>",
        ))
    figt.add_vline(x=top["base"], line_color="#15222a", line_width=1,
                   annotation_text=f"base {money(top['base'])}", annotation_position="top")
    figt.update_layout(
        barmode="overlay", height=60 + 45 * len(tornado), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="3-year net value ($)", tickformat="$,.0f"),
        yaxis=dict(autorange="reversed", title=""), legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(figt, width="stretch")
    st.caption(md(
        f"Each bar moves one assumption at a time; longest bar = biggest risk. "
        f"**{top['assumption'].replace(' ($)', '')}** matters most: it moves 3-year value by "
        f"{money(top['swing'])}. Agree this number with finance first."
    ))

    st.caption(
        f"Model trained on {res['n_train']:,} records and tested on {res['n_test']:,} records it never saw "
        f"({int(y_test.sum())} real failures). Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0). "
        "Dollar results scale the test-set rates to your plant's failures per year."
    )
    st.session_state["value"] = {**r, "model": model_name, "band": band, "cases": cases, "costs": costs,
                                 "top_driver": top["assumption"].replace(" ($)", ""), "top_swing": top["swing"]}

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Are we ready to go live, or will this stall as a pilot?")
    st.write("Rate each statement from 1 (not at all) to 5 (fully true). "
             "Pre-filled with an example pilot plant; change any answer.")
    EXAMPLE_PILOT = {"Data": [4, 4], "Workflow fit": [2, 2], "ROI clarity": [4, 3],
                     "User trust": [3, 2], "Governance & risk": [3, 3]}

    dim_scores = {}
    cols3 = st.columns(2)
    for i, (dim, qs) in enumerate(DIMENSIONS.items()):
        with cols3[i % 2]:
            st.markdown(f"**{dim}**")
            vals = [st.slider(q, 1, 5, EXAMPLE_PILOT[dim][j], key=f"rd_{dim}_{j}") for j, q in enumerate(qs)]
            dim_scores[dim] = sum(vals) / len(vals)

    v, weakest = verdict(dim_scores)
    avg = sum(dim_scores.values()) / len(dim_scores)
    msg = f"**Verdict: {v}** · average readiness {avg:.1f} / 5"
    {"Go": st.success, "Fix first": st.warning, "Not ready": st.error}[v](msg)

    figr = px.bar(
        pd.DataFrame({"Dimension": list(dim_scores), "Score": list(dim_scores.values())}),
        x="Score", y="Dimension", orientation="h", range_x=[0, 5], height=280, text="Score",
    )
    figr.update_traces(marker_color=["#b4322c" if s < 2.5 else "#a25f00" if s < 3.5 else "#1d7a45"
                                     for s in dim_scores.values()], texttemplate="%{text:.1f}")
    figr.add_vline(x=2.5, line_dash="dot", line_color="#8a979c")
    figr.add_vline(x=3.5, line_dash="dot", line_color="#8a979c")
    figr.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
    st.plotly_chart(figr, width="stretch")

    st.markdown("**Top 3 actions before launch**")
    for dim, sc in weakest:
        st.markdown(f"- **{dim}** ({sc:.1f}): {ACTIONS[dim]}")
    st.caption("Rules: Go = average ≥ 3.5 and no dimension below 2.5 · Fix first = any dimension below 2.5 "
               "(or average 2.5–3.5) · Not ready = average below 2.5.")
    st.session_state["readiness"] = {"verdict": v, "avg": avg, "weakest": weakest}

# ---------------------------------------------------------------- Tab 5
# Runs before Tab 4 so the business case summary can include value delivered to date.
GAP_ACTIONS = {
    "Failures (volume)": "The plant had a different number of failures than the business case assumed. "
                         "This is not the model's fault; update failures per year in the Prove value tab.",
    "Model recall": "The model is catching fewer failures than it did in testing. Retrain it on recent data.",
    "Flagged failures acted on": "The model warned about failures the crew did not act on. "
                                 "Fix the workflow and build trust (see the Readiness tab).",
    "False alarms": "There were more false alarms than in testing. Review the alert threshold.",
}

with tab5:
    st.subheader("After go-live: is the AI delivering the value we promised?")
    st.write(
        "Business cases are approved on a forecast, but few teams check the forecast once the model is live. "
        "Upload monthly counts from your maintenance system to compare **actual savings with the forecast**, "
        "see whether the crew acts on alerts, and catch the model drifting before value drops."
    )
    val = st.session_state.get("value", {})
    if not val:
        st.info("Open the Prove value tab first.")
    else:
        sample = pd.read_csv("data/sample_actuals.csv")
        u1, u2 = st.columns([3, 1])
        upload = u1.file_uploader("Monthly actuals (CSV, one row per month)", type="csv", key="actuals_upload")
        u2.download_button("Download template (sample data)", sample.to_csv(index=False),
                           file_name="monthly_actuals_template.csv", key="dl_template")
        actuals = pd.read_csv(upload) if upload is not None else sample
        source = "uploaded file" if upload is not None else "simulated sample"
        if upload is None:
            st.caption("Showing a **simulated** 12-month rollout, not real plant data: adoption is slow at first, "
                       "then the model starts to drift in month 9. Upload your own file to replace it.")
        with st.expander("What goes in the file"):
            st.markdown(
                "- `month`: e.g. 2026-01\n"
                "- `alerts_raised`: alerts the model sent\n"
                "- `failures_flagged`: failures the model warned about in advance (acted on or not)\n"
                "- `failures_prevented`: flagged failures the crew fixed in time\n"
                "- `failures_missed`: unplanned failures (not flagged, or flagged but ignored)\n"
                "- `false_alarms`: inspections that found nothing wrong\n\n"
                "Dollars come from these counts and the costs in the Prove value tab, using the same formula as the forecast."
            )

        errors = validate(actuals)
        if errors:
            st.error("This file can't be used:\n\n" + "\n".join(f"- {e}" for e in errors))
            st.session_state.pop("tracking", None)
        else:
            c = val["costs"]
            t = track(actuals, val, c)
            gap = explain_gap(t, val, c)
            last = t.iloc[-1]
            n = len(t)
            delivered, promised = last["cum_actual"], last["cum_forecast"]
            pct = delivered / promised if promised > 0 else np.nan
            recall_now = t["recall_3m"].dropna().iloc[-1] if t["recall_3m"].notna().any() else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Savings after {n} months", money(delivered), f"{money(delivered - promised)} vs forecast")
            m2.metric("Share of forecast delivered", f"{pct*100:.0f}%" if np.isfinite(pct) else "n/a")
            m3.metric("Alerts acted on (last month)", f"{last['adoption']*100:.0f}%" if pd.notna(last["adoption"]) else "n/a",
                      help=f"Warning below {ADOPTION_FLOOR:.0%}.")
            m4.metric(f"Recall (last {DRIFT_WINDOW} months)", f"{recall_now*100:.0f}%" if np.isfinite(recall_now) else "n/a",
                      f"{(recall_now - val['recall'])*100:+.0f} pts vs launch" if np.isfinite(recall_now) else None,
                      help=f"Launch recall {val['recall']*100:.0f}% on test data. Warning when {DRIFT_DROP*100:.0f}+ points lower.")

            low_months = t.loc[t["adoption_alert"], "month"].tolist()
            if last["adoption_alert"]:
                st.error(f"**Adoption alert:** the crew acted on only {last['adoption']*100:.0f}% of alerts last month "
                         f"(target {ADOPTION_FLOOR:.0%}). Value is being left on the table.")
            elif low_months:
                st.info(f"Adoption was below {ADOPTION_FLOOR:.0%} in {', '.join(low_months)} and has since recovered.")
            drift_months = t.loc[t["drift_alert"], "month"].tolist()
            if last["drift_alert"]:
                st.error(f"**Drift alert:** recall over the last {DRIFT_WINDOW} months is {recall_now*100:.0f}%, "
                         f"{(val['recall'] - recall_now)*100:.0f} points below launch. Retrain the model before value drops further "
                         f"(first flagged {drift_months[0]}).")
            elif drift_months:
                st.warning(f"Recall dipped below the drift line in {', '.join(drift_months)} but is back above it.")

            lo, hi = val["band"]["annual_net"]
            k = np.arange(1, n + 1)
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=t["month"], y=hi / 12 * k, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            figc.add_trace(go.Scatter(x=t["month"], y=lo / 12 * k, mode="lines", fill="tonexty", fillcolor="rgba(15,109,116,0.12)",
                                      line=dict(width=0), name="Forecast range (model uncertainty)", hoverinfo="skip"))
            figc.add_trace(go.Scatter(x=t["month"], y=t["cum_forecast"], name="Forecast", mode="lines", line=dict(color="#0f6d74", dash="dash")))
            figc.add_trace(go.Scatter(x=t["month"], y=t["cum_actual"], name="Actual", mode="lines+markers",
                                      line=dict(color="#15222a")))
            figc.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Cumulative net savings since go-live",
                               yaxis=dict(tickformat="$,.0f", title=""), xaxis_title="", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(figc, width="stretch")

            g1, g2 = st.columns(2)
            figa = go.Figure(go.Bar(x=t["month"], y=t["adoption"] * 100,
                                    marker_color=["#b4322c" if a else "#1d7a45" for a in t["adoption_alert"]]))
            figa.add_hline(y=ADOPTION_FLOOR * 100, line_dash="dot", line_color="#8a979c",
                           annotation_text=f"{ADOPTION_FLOOR:.0%} target", annotation_position="top left")
            figa.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Alerts acted on (%)",
                               yaxis=dict(range=[0, 105], title=""), xaxis_title="")
            g1.plotly_chart(figa, width="stretch")

            figd = go.Figure()
            figd.add_trace(go.Scatter(x=t["month"], y=t["recall_3m"] * 100, mode="lines+markers", name="Recall (3-month)",
                                      line=dict(color="#15222a")))
            figd.add_hline(y=val["recall"] * 100, line_color="#0f6d74", annotation_text="launch",
                           annotation_position="top left")
            figd.add_hline(y=(val["recall"] - DRIFT_DROP) * 100, line_dash="dot", line_color="#b4322c",
                           annotation_text="drift alert", annotation_position="bottom left")
            figd.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                               title=f"Failures caught, rolling {DRIFT_WINDOW} months (%)",
                               yaxis=dict(range=[0, 105], title=""), xaxis_title="")
            g2.plotly_chart(figd, width="stretch")

            st.markdown("**Why actual differs from the forecast**")
            figw = go.Figure(go.Waterfall(
                x=["Forecast"] + gap["cause"].tolist() + ["Actual"],
                measure=["absolute"] + ["relative"] * len(gap) + ["total"],
                y=[promised] + gap["effect"].tolist() + [0],
                text=[money(promised)] + [money(e) for e in gap["effect"]] + [money(delivered)], textposition="outside",
                increasing=dict(marker_color="#1d7a45"), decreasing=dict(marker_color="#b4322c"),
                totals=dict(marker_color="#15222a"), connector=dict(line=dict(color="#8a979c")),
            ))
            figw.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(tickformat="$,.0f", title=""),
                               showlegend=False)
            st.plotly_chart(figw, width="stretch")

            fmt = {"Failures (volume)": lambda x: f"{x:.0f} failures", "Model recall": lambda x: f"{x*100:.0f}%",
                   "Flagged failures acted on": lambda x: f"{x*100:.0f}%", "False alarms": lambda x: f"{x:.0f}"}
            st.dataframe(pd.DataFrame({
                "Cause": gap["cause"],
                "Forecast assumed": [fmt[cs](a) for cs, a in zip(gap["cause"], gap["assumed"])],
                "Actual": [fmt[cs](a) for cs, a in zip(gap["cause"], gap["actual"])],
                "Effect on savings": [money(e) for e in gap["effect"]],
            }), hide_index=True, width="stretch")
            worst = gap.loc[gap["effect"].idxmin()]
            if worst["effect"] < 0:
                st.markdown(md(f"**Biggest shortfall: {worst['cause']}** ({money(worst['effect'])}). "
                               f"{GAP_ACTIONS[worst['cause']]}"))
            st.caption(md(
                "Each cause is swapped from its forecast value to its actual value one at a time, so the effects add up "
                "exactly to the gap. The forecast assumes every alert is acted on. Dollar costs per failure can't be "
                f"checked from counts: the biggest cost driver in Prove value is **{val['top_driver']}**, so check it "
                "against real repair invoices."
            ))

            with st.expander("Monthly data"):
                st.dataframe(t[COLUMNS + ["recall", "adoption", "actual", "forecast"]].round(2), hide_index=True, width="stretch")

            st.session_state["tracking"] = {
                "source": source, "months": n, "first": t["month"].iloc[0], "last": last["month"],
                "delivered": delivered, "promised": promised, "pct": pct,
                "adoption_last": last["adoption"], "low_adoption_months": low_months,
                "recall_now": recall_now, "drift_now": bool(last["drift_alert"]),
                "worst_cause": worst["cause"] if worst["effect"] < 0 else None, "worst_effect": worst["effect"],
            }

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.subheader("Business case summary")
    val = st.session_state.get("value", {})
    rd = st.session_state.get("readiness", {})
    tk = st.session_state.get("tracking")
    tracking_md = ""
    if tk:
        pct_txt = f"{tk['pct']*100:.0f}% of forecast" if np.isfinite(tk["pct"]) else "forecast not positive"
        tracking_md = (
            f"\n## Value delivered after go-live ({tk['source']}, {tk['first']} to {tk['last']})\n"
            f"- Net savings to date: {money(tk['delivered'])} vs {money(tk['promised'])} forecast ({pct_txt})\n"
            + (f"- Biggest shortfall: {tk['worst_cause']} ({money(tk['worst_effect'])}). "
               f"{GAP_ACTIONS[tk['worst_cause']]}\n" if tk["worst_cause"] else "")
            + (f"- Adoption below {ADOPTION_FLOOR:.0%} in: {', '.join(tk['low_adoption_months'])}\n"
               if tk["low_adoption_months"] else f"- Adoption at or above {ADOPTION_FLOOR:.0%} every month\n")
            + ("- Drift alert: recall is "
               f"{tk['recall_now']*100:.0f}% over the last {DRIFT_WINDOW} months, {DRIFT_DROP*100:.0f}+ points below launch. Retrain.\n"
               if tk["drift_now"] else "- No drift alert in the latest month\n")
        )
    if not val:
        st.info("Open the Prove value tab first.")
    else:
        summary = f"""# AI Business Case: Predictive Maintenance

**Use case:** Predictive maintenance (manufacturing). Top-ranked use case in the Prioritize tab: {st.session_state.get('top_use_case', 'n/a')} ({st.session_state.get('top_quadrant', 'n/a')}).

## Value
- Annual net savings: {money(val['annual_net'])} (90% range from model uncertainty: {money_range(val['band']['annual_net'])})
- Payback: {months(val['payback_months'])}
- 3-year ROI: {val['roi_3y']*100:,.0f}%
- First-year value (after deployment cost): {money(val['first_year'])}
- Model: {val['model']}, alert threshold {val['threshold']:.2f}
- {plain_language(val)}
- Per year: ~{val['caught']:.0f} failures caught, ~{val['missed']:.0f} missed, ~{val['false_alarms']:.0f} false alarms

## Range of outcomes (cost assumptions)
| Case | Annual net savings | 3-year net value | Payback |
|---|---|---|---|
""" + "".join(f"| {name} | {money(c['annual_net'])} | {money(c['three_year'])} | {months(c['payback_months'])} |\n"
              for name, c in val["cases"].items()) + f"""
- Biggest driver: {val['top_driver']} (moves 3-year value by {money(val['top_swing'])}). Agree this number with finance first.

## Readiness
- Verdict: {rd.get('verdict', 'n/a')} (average {rd.get('avg', 0):.1f} / 5)
""" + "".join(f"- Fix: {d} ({s:.1f}): {ACTIONS[d]}\n" for d, s in rd.get("weakest", [])) + tracking_md + """
## Key assumptions
- Plant costs as entered in the Prove value tab (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
        st.markdown(md(summary))
        st.download_button("Download summary (.md)", summary, file_name="ai_business_case.md", key="dl_summary")
