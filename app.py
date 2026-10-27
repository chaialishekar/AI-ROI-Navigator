"""AI Use-Case ROI Navigator

An independent concept project: helps a company decide which AI use case to deploy,
prove its value in dollars, and check whether it is ready for production.

Overview          - what the project shows, headline results, and the steps to click into
Plan the project
  1 Pick a use case   - score use cases on value vs feasibility
  2 Prove the value   - tune a real predictive maintenance model for dollars, not accuracy
  3 Check readiness   - pilot-to-production readiness check
  4 Business case     - one-page summary to share with finance
After launch
  5 Track the value   - forecast vs actual savings, adoption and drift (fed by the S3 -> Databricks pipeline)
Generative AI
  6 GenAI pilot       - measured Bedrock pilot of an AI work-order writer, and its ROI across five cost categories
  7 AI analyst agents - intake, memo and red-team agents on Bedrock, graded automatically (v1 vs v2 vs v3)

Every page opens with its question and the answer, then the detail, then Back / Next links.
"""

import json
import os

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
from src.genai import COST_LABELS, GenAIInputs
from src.genai import evaluate as genai_evaluate
from src.tracking import ADOPTION_FLOOR, COLUMNS, DRIFT_DROP, DRIFT_WINDOW, explain_gap, track, validate
from src.agents import ASSUMPTIONS as AGENT_ASSUMPTIONS, INTAKE_FIELDS, run_live
from src.agents import MODEL_ID as AGENT_MODEL_ID
from agents.grade import scorecard
from src.workorder import MODEL_ID, MODES, grade, parse, summary as pilot_summary
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

st.set_page_config(page_title="AI Use-Case ROI Navigator", page_icon="📈", layout="wide")

QUADRANT_COLORS = {
    "Quick win": "#1d7a45",
    "Big bet": "#0f6d74",
    "Fill-in": "#a25f00",
    "Deprioritize": "#8a979c",
}


@st.cache_data
def load_use_cases() -> pd.DataFrame:
    return pd.read_csv("data/use_cases.csv")



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


# ---------------------------------------------------------------- Page 1
def page_pick():
    answer = page_header(1, "Which AI project should we do first?",
                         "12 ideas, each scored 1–5 on **value** (money, strategy, scale) and **feasibility** "
                         "(data, effort, risk, speed). High on both = **quick win**.")

    industries = sorted(load_use_cases()["industry"].unique())
    with st.expander("Change the weights or industries"):
        picked = st.multiselect("Industries", industries, default=industries, key="industries")
        w1, w2 = st.columns(2)
        w1.markdown("**What counts as value**")
        value_w = {k: w1.slider(lbl, 0.0, 1.0, DEFAULT_VALUE_WEIGHTS[k], 0.05, key=f"vw_{k}")
                   for k, lbl in VALUE_CRITERIA.items()}
        w2.markdown("**What counts as feasible**")
        feas_w = {k: w2.slider(lbl, 0.0, 1.0, DEFAULT_FEAS_WEIGHTS[k], 0.05, key=f"fw_{k}")
                  for k, lbl in FEASIBILITY_CRITERIA.items()}
        st.caption("Weights in each group are normalized.")

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
    top = scored.iloc[0]
    n_wins = int((scored["quadrant"] == "Quick win").sum())
    pdm = scored[scored["use_case"] == "Predictive maintenance"]
    answer.markdown(md(
        f"Start with **{top['use_case']}** ({top['industry']}): it scores highest on value × feasibility. "
        f"{n_wins} of {len(scored)} ideas are quick wins."
        + (f" The rest of this app follows **predictive maintenance** (ranked #{int(pdm['rank'].iloc[0])}, "
           f"{pdm['quadrant'].iloc[0].lower()}), because a public machine dataset lets us prove its value with a real model."
           if not pdm.empty else "")))

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
    with st.expander("Ranked list with next steps"):
        st.dataframe(
            scored[["rank", "use_case", "industry", "value", "feasibility", "priority", "quadrant", "next_step"]],
            hide_index=True,
            width="stretch",
        )
    nav_footer("pick")

# ---------------------------------------------------------------- Page 2
def page_value():
    answer = page_header(2, "What is it worth in dollars?",
                         "A model trained on 10,000 machine records warns before a machine fails. We turn its catches, "
                         "misses and false alarms into dollars for a plant with 40 breakdowns a year.")

    res = get_model_results()
    y_test = res["y_test"]

    with st.expander("Your plant's costs (edit to match your operation)"):
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
    st.caption(plain_language(r))

    with st.expander("Failures caught, missed and false alarms per year"):
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

    with st.expander("Model comparison (each at its most profitable threshold)"):
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
    st.markdown("#### How sure are we?")
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
    answer.markdown(md(
        f"**{money(r['annual_net'])} a year**, paying back in **{months(r['payback_months'])}**. "
        f"Even with every cost assumption at its pessimistic end it saves {money(low['annual_net'])} a year. "
        + (f"Setting alerts for dollars instead of accuracy is worth **{money(gap)} more a year**." if gap > 1 else "")))
    nav_footer("value")

# ---------------------------------------------------------------- Page 3
EXAMPLE_PILOT = {"Data": [4, 4], "Workflow fit": [2, 2], "ROI clarity": [4, 3],
                 "User trust": [3, 2], "Governance & risk": [3, 3]}


def page_ready():
    answer = page_header(3, "Are we ready to go live?",
                         "Most AI pilots stall because of people and process, not the model. Rate 10 statements "
                         "from 1 (not at all) to 5 (fully true); it's pre-filled with an example plant.")

    dim_scores = {}
    cols3 = st.columns(2)
    for i, (dim, qs) in enumerate(DIMENSIONS.items()):
        with cols3[i % 2]:
            st.markdown(f"**{dim}**")
            vals = [st.slider(q, 1, 5, EXAMPLE_PILOT[dim][j], key=f"rd_{dim}_{j}") for j, q in enumerate(qs)]
            dim_scores[dim] = sum(vals) / len(vals)

    v, weakest = verdict(dim_scores)
    avg = sum(dim_scores.values()) / len(dim_scores)
    weak = ", ".join(f"**{d.lower()}**" for d, _ in weakest)
    answer.markdown({"Go": f"**Go** (average {avg:.1f} / 5). Weakest areas to keep an eye on: {weak}.",
                     "Fix first": f"**Not yet: fix first** (average {avg:.1f} / 5). Before launch, work on {weak}.",
                     "Not ready": f"**Not ready** (average {avg:.1f} / 5). Start with {weak}."}[v])

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
    nav_footer("ready")

# ---------------------------------------------------------------- Page 5
SHORTFALL = {"Failures (volume)": "the plant had fewer breakdowns to prevent than assumed",
             "Model recall": "the model caught fewer failures than in testing",
             "Flagged failures acted on": "the crew ignored some correct alerts",
             "False alarms": "more false alarms than expected"}
GAP_ACTIONS = {
    "Failures (volume)": "The plant had a different number of failures than the business case assumed. "
                         "This is not the model's fault; update failures per year on the Prove the value page.",
    "Model recall": "The model is catching fewer failures than it did in testing. Retrain it on recent data.",
    "Flagged failures acted on": "The model warned about failures the crew did not act on. "
                                 "Fix the workflow and build trust (see Check readiness).",
    "False alarms": "There were more false alarms than in testing. Review the alert threshold.",
}

@st.cache_data
def load_pipeline() -> dict | None:
    """Gold export from the S3 -> Databricks pipeline (pipeline/run.py), if it has been run."""
    try:
        gold = pd.read_csv("data/pipeline/tracked_value.csv")
        with open("data/pipeline/run_log.json") as f:
            log = json.load(f)
    except FileNotFoundError:
        return None
    P = log["parameters"]
    c = Costs(cost_missed=P["cost_missed"], cost_caught=P["cost_caught"], cost_false_alarm=P["cost_false_alarm"],
              running_cost=P["running_cost"])
    t = track(gold[COLUMNS], {"annual_net": P["annual_net"], "recall": P["launch_recall"]}, c)
    matches = np.allclose(gold["cum_actual"], t["cum_actual"]) and (gold["drift_alert"] == t["drift_alert"]).all()
    return {"gold": gold, "log": log, "matches": bool(matches)}


def page_track():
    answer = page_header(5, "After launch: is it delivering what we promised?",
                         "Few teams check the forecast once the model is live. Monthly counts from the maintenance "
                         "system are turned into dollars and compared with the forecast, alongside two early warnings: "
                         "the crew ignoring alerts, and the model getting worse.")
    val = get_value()
    sample = pd.read_csv("data/sample_actuals.csv")
    pipe = load_pipeline()
    with st.expander("Data source: pipeline, sample or your own file"):
        u1, u2 = st.columns([3, 1])
        upload = u1.file_uploader("Monthly actuals (CSV, one row per month)", type="csv", key="actuals_upload")
        u2.download_button("Download template (sample data)", sample.to_csv(index=False),
                           file_name="monthly_actuals_template.csv", key="dl_template")
        use_pipe = upload is None and pipe is not None and st.radio(
            "Data source", ["Pipeline: S3 → Databricks gold table", "Bundled sample file"], horizontal=True,
            key="actuals_source").startswith("Pipeline")
        st.markdown(
            "**File format**, one row per month:\n"
            "- `month`: e.g. 2026-01\n"
            "- `alerts_raised`: alerts the model sent\n"
            "- `failures_flagged`: failures the model warned about in advance (acted on or not)\n"
            "- `failures_prevented`: flagged failures the crew fixed in time\n"
            "- `failures_missed`: unplanned failures (not flagged, or flagged but ignored)\n"
            "- `false_alarms`: inspections that found nothing wrong\n\n"
            "Dollars come from these counts and the costs on the Prove the value page, using the same formula as the forecast."
        )
    if upload is not None:
        actuals, source = pd.read_csv(upload), "uploaded file"
    elif use_pipe:
        actuals, source = pipe["gold"][COLUMNS], "S3 → Databricks pipeline"
    else:
        actuals, source = sample, "simulated sample"
    if upload is None and not use_pipe:
        st.caption("Showing a **simulated** 12-month rollout, not real plant data: adoption is slow at first, "
                   "then the model starts to drift in month 9. Upload your own file to replace it.")
    if use_pipe:
        log = pipe["log"]
        st.caption(md(
            f"Loaded from the **pipeline** (the same simulated 12-month rollout, not real plant data): monthly files land in an encrypted S3 bucket (us-east-2), are synced to "
            f"Databricks and flow through bronze → silver → gold tables. Last refresh {log['refreshed_at'][:16].replace('T', ' ')} UTC; "
            f"{log['bronze_rows']} rows received, {log['silver_rows']} months passed the checks. "
            + ("Gold matches this page's formula ✅" if pipe["matches"] else "⚠️ Gold differs from this page's formula")))
        for q in log["quarantined"]:
            st.warning(md(f"**Quarantined** `{q['file']}` ({q['month']}): {q['reason']}. Ask the plant to fix and resend it."))

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

        tk = st.session_state["tracking"] = summarize_tracking(t, gap, source)
        answer.markdown(md(
            f"**{pct*100:.0f}% of the forecast delivered** after {n} months ({money(delivered)} of {money(promised)}). "
            + (f"Biggest shortfall: {SHORTFALL[tk['worst_cause']]}. " if tk["worst_cause"] else "")
            + ("**Drift alert:** the model is catching fewer failures; retrain it." if tk["drift_now"] else "")))
    nav_footer("track")

# ---------------------------------------------------------------- Page 4
def page_case():
    answer = page_header(4, "The one-page business case",
                         "Everything from steps 1–3, plus value delivered after launch, in a summary you can download "
                         "and send to finance. It updates when you change any input on the earlier pages.")
    val = get_value()
    rd = get_readiness()
    answer.markdown(md(
        f"**{money(val['annual_net'])} a year, {months(val['payback_months'])} payback.** Readiness says "
        f"**{rd['verdict'].lower()}**" + (f", so fix {rd['weakest'][0][0].lower()} before launch" if rd["verdict"] != "Go" else "")
        + ". Download the one-pager below."))
    tk = st.session_state.get("tracking") or default_tracking()
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
    summary = f"""# AI Business Case: Predictive Maintenance

**Use case:** Predictive maintenance (manufacturing). Top-ranked use case on the Pick a use case page: {st.session_state.get('top_use_case', 'n/a')} ({st.session_state.get('top_quadrant', 'n/a')}).

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
- Plant costs as entered on the Prove the value page (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
    st.download_button("Download summary (.md)", summary, file_name="ai_business_case.md", key="dl_summary",
                       type="primary")
    with st.container(border=True):
        st.markdown(md(summary))
    nav_footer("case")

# ---------------------------------------------------------------- Page 6
@st.cache_data
def load_pilot() -> dict:
    cases = pd.read_csv("pilot/pilot_cases.csv")
    runs = {v: grade(pd.read_csv(f"pilot/pilot_results_v{v}.csv")) for v in (1, 2)}
    return {"cases": cases, "runs": runs, "summary": {v: pilot_summary(g) for v, g in runs.items()}}


def show_draft(reply: str):
    d = parse(reply)
    if not d:
        st.code(reply)
        return
    esc = lambda t: md(str(t)).replace("*", "\\*")      # model text: no accidental markdown or math
    st.markdown(f"**{esc(d.get('title', ''))}** · {d.get('priority', '')} · cause: **{d.get('failure_mode', '')}**")
    st.caption(esc(d.get("reason", "")))
    st.markdown("\n".join(f"{i}. {esc(step)}" for i, step in enumerate(d.get("steps", []), 1)))
    if d.get("parts"):
        st.caption("Parts: " + ", ".join(d["parts"]))


def page_pilot():
    answer = page_header(6, "Does a GenAI assistant pay for itself?",
                         "When the model raises an alert, a small language model (Amazon Nova Micro on Bedrock) drafts "
                         "the work order. We ran 30 real alerts through it twice and measured cost, speed and whether "
                         "each draft named the right cause.")
    pilot = load_pilot()
    s1, s2 = pilot["summary"][1], pilot["summary"][2]
    answer.markdown(md(
        f"Only if the AI is kept away from the diagnosis. Writing the work order costs **${s2['cost_per_task']*1000:.2f} "
        f"per 1,000 drafts**, but when the AI also diagnosed the fault it was right only "
        f"**{s1['correct_on_real_failures']:.0%}** of the time on real failures. Letting simple rules diagnose and the AI "
        f"only write raised that to **{s2['correct_on_real_failures']:.0%}**. Review time, not the model fee, decides the ROI."))

    st.markdown("**Pilot results** (30 alerts each, run on Bedrock in us-east-2)")
    st.dataframe(pd.DataFrame({
        "Version 1: AI diagnoses and writes": [
            f"{s1['correct_mode_rate']:.0%}", f"{s1['correct_on_real_failures']:.0%}",
            f"{s1['false_alarm_handled_rate']:.0%}", f"${s1['cost_per_task']*1000:.3f}", f"{s1['median_latency_s']:.1f} s"],
        "Version 2: rule check diagnoses, AI writes": [
            f"{s2['correct_mode_rate']:.0%}", f"{s2['correct_on_real_failures']:.0%}",
            f"{s2['false_alarm_handled_rate']:.0%}", f"${s2['cost_per_task']*1000:.3f}", f"{s2['median_latency_s']:.1f} s"],
    }, index=["Right cause (all 30)", "Right cause (24 real failures)", "False alarm recognised (6)",
              "Model cost per 1,000 drafts", "Median response time"]), width="stretch")
    st.warning(
        "**Version 1 failed:** the model said \"overstrain\" 23 times, 17 of them when the readings were below the limit. "
        "It writes fluently but cannot be trusted to compare numbers. **Version 2** moves the diagnosis into a few lines of "
        "code (always right on the 24 real failures) and leaves the model to write. It kept the given diagnosis in all 30 "
        "drafts, but 4 drafts still mention computer parts such as \"thermal grease on the CPU\", so a human review stays."
    )
    st.caption(f"Model {MODEL_ID}, temperature 0, max 500 output tokens. The 5 false alarms version 2 still flags are "
               "machines with tools at 208-246 min of wear, where the manual says to replace the tool anyway.")

    with st.expander("Compare drafts for one alert"):
        cases = pilot["cases"]
        pick = st.selectbox("Alert", cases["case_id"], key="pilot_case",
                            format_func=lambda c: f"{c} (true cause: {MODES[cases.set_index('case_id').loc[c, 'true_mode']]})")
        d1, d2 = st.columns(2)
        for col, v in ((d1, 1), (d2, 2)):
            row = pilot["runs"][v].set_index("case_id").loc[pick]
            col.markdown(f"**Version {v}** {'✅' if row['correct_mode'] else '❌'}")
            with col:
                show_draft(row["reply"])

    st.divider()
    st.markdown("**Is it worth it? ROI across the five cost categories**")
    st.caption("Cost categories from Dogra, \"How To Measure AI ROI In A Small Business\" (Forbes, 2026). "
               "Model cost per draft and wrong-cause rate come from the pilot; the rest are editable assumptions.")
    a1, a2, a3 = st.columns(3)
    tasks = a1.number_input("Work orders per year", 100, 1_000_000, 2_000, 100, key="g_tasks")
    saved = a1.number_input("Minutes saved per work order (before review)", 1.0, 120.0, 15.0, 1.0, key="g_saved")
    rate = a1.number_input("Loaded hourly cost of a planner ($)", 10, 500, 55, 5, key="g_rate")
    review = {1: a2.number_input("Review minutes per draft, version 1", 0.0, 60.0, 12.0, 0.5, key="g_rev1",
                                 help="Two in three drafts had the wrong cause, so every draft must be re-diagnosed."),
              2: a2.number_input("Review minutes per draft, version 2", 0.0, 60.0, 4.0, 0.5, key="g_rev2",
                                 help="Cause is reliable; review checks the steps and parts.")}
    missed = a2.slider("Share of wrong drafts that slip past review", 0.0, 1.0, 0.1, 0.05, key="g_missed")
    cost_err = a3.number_input("Cost when a wrong work order reaches the floor ($)", 0, 100_000, 150, 25, key="g_err")
    setup_h = a3.number_input("Setup hours (integration, prompts, rule check)", 0, 2_000, 40, 5, key="g_setup")
    redeploy = a3.slider("Share of saved hours that turn into cash", 0.0, 1.0, 0.5, 0.05, key="g_redeploy",
                         help="Saved hours only become money if they go to other work or delay a hire.")
    year_one = st.toggle("Include one-time setup and training (first year)", True, key="g_year1")

    out = {}
    for v, sm in pilot["summary"].items():
        g = GenAIInputs(tasks_per_year=tasks, minutes_saved_per_task=saved, hourly_cost=rate,
                        model_cost_per_task=sm["cost_per_task"], setup_hours=setup_h,
                        review_minutes_per_task=review[v], error_rate=(1 - sm["correct_mode_rate"]) * missed,
                        cost_per_error=cost_err, share_redeployed=redeploy)
        out[v] = genai_evaluate(g, year_one=year_one)

    k1, k2, k3, k4 = st.columns(4)
    r2 = out[2]
    k1.metric("Net value, version 2", money(r2["net"]), f"{money(r2['net'] - out[1]['net'])} vs version 1")
    k2.metric("ROI, version 2", f"{r2['roi']*100:,.0f}%", f"{(r2['roi'] - out[1]['roi'])*100:+,.0f} pts vs version 1")
    k3.metric("Cash ROI, version 2", f"{r2['cash_roi']*100:,.0f}%",
              help="Counts only the share of saved hours that turns into cash.")
    k4.metric("Break-even review time", f"{r2['break_even_review_minutes']:.1f} min",
              help="If reviewing a draft takes longer than this, the assistant loses money.")

    cost_rows = [{"Version": f"Version {v}", "Category": COST_LABELS[k], "Cost": c}
                 for v, r in out.items() for k, c in r["costs"].items()]
    figg = px.bar(pd.DataFrame(cost_rows), x="Cost", y="Version", color="Category", orientation="h", height=260,
                  color_discrete_sequence=["#0f6d74", "#8a979c", "#a25f00", "#1d7a45", "#b4322c"])
    for v, r in out.items():
        figg.add_trace(go.Scatter(x=[r["gain"]], y=[f"Version {v}"], mode="markers", name="Value of time saved" if v == 1 else None,
                                  showlegend=v == 1, marker=dict(symbol="line-ns", size=28, line=dict(width=3, color="#15222a"))))
    figg.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(tickformat="$,.0f", title="Cost per year"),
                       yaxis_title="", legend=dict(orientation="h", y=-0.35), barmode="stack")
    st.plotly_chart(figg, width="stretch")

    usage = r2["costs"]["usage"]
    st.markdown(md(
        f"**The model fee is not the cost.** Bedrock charges for {tasks:,} drafts come to **${usage:,.2f} a year**; "
        f"review time is **{money(r2['costs']['review'])}**. Version 1 costs almost the same to run but needs "
        f"{review[1]:.0f}-minute reviews, which "
        + ("wipes out the gain." if out[1]["net"] <= 0 else f"cuts net value to {money(out[1]['net'])}.")
        + f" On cash, only {redeploy:.0%} of the saved hours count, giving a cash ROI of {r2['cash_roi']*100:,.0f}%."
    ))
    st.caption("Black tick = value of time saved. Training: 2 hours for 6 planners. Error rate = wrong-cause rate from "
               "the pilot × share that slips past review.")
    nav_footer("pilot")


# ---------------------------------------------------------------- Tab 7
@st.cache_data
def load_agents() -> dict:
    with open("agents/eval_cases.json") as f:
        cases = json.load(f)
    with open("agents/results.json") as f:
        runs = json.load(f)
    return {"cases": cases, "runs": runs, "scores": {v: scorecard(r, cases) for v, r in runs.items()}}


def aws_ready() -> bool:
    """Live mode only where AWS credentials exist (local use), so visitors cannot spend the owner's credits."""
    try:
        import boto3
        return boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).get_credentials() is not None
    except Exception:
        return False


def agent_reply(version: str, case_id: str) -> str:
    row = next(r for r in load_agents()["runs"][version] if r["id"] == case_id)
    return row.get("final", row["reply"])


def show_objections(reply: str):
    d = parse(reply) or {}
    esc = lambda t: md(str(t)).replace("*", "\\*")
    for o in d.get("objections", []):
        icon = {"high": "🔴", "medium": "🟠", "low": "⚪"}.get(o.get("severity"), "⚪")
        line = f"{icon} **{esc(o.get('assumption'))}** ({o.get('severity', 'n/a')}): {esc(o.get('issue') or o.get('why') or '')}"
        if o.get("issue") and o.get("why"):
            line += f"  \n{esc(o['why'])}"
        st.markdown(line)
        if o.get("test"):
            st.caption("30-day test: " + esc(o["test"]))
    if not d.get("objections"):
        st.caption("No objections.")
    st.markdown(f"Verdict: **{d.get('verdict', 'n/a')}**")


def page_agents():
    answer = page_header(7, "Can AI agents do the analyst's paperwork?",
                         "Three small agents on Amazon Bedrock (Nova Lite): **Intake** turns a manager's paragraph into "
                         "calculator inputs, **Memo** writes the CFO memo, **Red team** plays the skeptical CFO. Each is "
                         "graded automatically on cases with known answers, including business cases with a planted flaw.")
    AGENTS = load_agents()
    sc = AGENTS["scores"]
    answer.markdown(md(
        f"Yes, once **code checks the numbers and the AI only writes**. Intake went from "
        f"{sc['v1']['intake_field_accuracy']:.0%} to {sc['v3']['intake_field_accuracy']:.0%} of fields right, memos stopped "
        f"inventing numbers, and the red team went from {sc['v1']['redteam_caught']} to {sc['v3']['redteam_caught']} of "
        f"{sc['v3']['redteam_flaws']} planted flaws caught, for about ${sc['v3']['total_cost']:.3f} a run."))
    col = lambda v: [
        f"{sc[v]['intake_field_accuracy']:.0%}", f"{sc[v]['intake_invented_fields']}",
        f"{sc[v]['memo_faithful']} of 2", f"{sc[v]['redteam_caught']} of {sc[v]['redteam_flaws']}",
        f"{sc[v]['redteam_caught_first']} of {sc[v]['redteam_flaws']}", f"{sc[v]['redteam_clean_high']}",
        f"${sc[v]['total_cost']:.4f}"]
    st.markdown("**Scorecard** (Nova Lite, temperature 0, run in us-east-2)")
    st.dataframe(pd.DataFrame({
        "v1: the model does everything": col("v1"),
        "v2: code converts and fills in numbers": col("v2"),
        "v3: code checklist judges, model explains": col("v3"),
    }, index=["Intake: fields right (8 descriptions)", "Intake: numbers invented when not stated",
              "Memo: memos with no invented numbers", "Red team: planted flaw raised as high/medium",
              "Red team: planted flaw listed first", "Red team: false 'high' alarms on the sound case",
              "Cost of the whole eval run"]), width="stretch")
    st.info(
        "**What changed.** v1 wrote arithmetic like `300 * 52` inside its JSON, invented a $5,000 figure in a memo, and gave "
        "nearly the same five objections to every case. **v2:** the model only quotes the numbers it was given and code "
        "converts weeks to years and checks the quote is really in the text; the memo uses placeholders like "
        "`{First-year ROI}` that code fills in. Intake and memo became exact. The red team still failed even when handed "
        "the ratios: it called \"15 of 20 minutes saved\" impossible in every case. **v3:** a code checklist with one "
        "plausibility rule per assumption decides what is wrong and how serious it is; the model writes the CFO's "
        "explanation and a 30-day test. Same lesson as the pilot: code does numbers and checks, the model does language."
    )
    st.caption(
        f"Model {AGENT_MODEL_ID}. Honest limits: the two held-out flaws (hourly cost $400, zero training) were caught "
        f"by v3, but they were known when the checklist was written, so this is not a blind test. In v3 the model "
        f"explained {sc['v3']['redteam_explained']} of {sc['v3']['redteam_flags']} flags; once it used the wrong "
        "assumption name, and code showed the flag without an explanation instead of guessing."
    )

    cases = {k: {c["id"]: c for c in v} for k, v in AGENTS["cases"].items()}
    with st.expander("Intake agent: paragraph in, calculator inputs out"):
        pick = st.selectbox("Description", list(cases["intake"]), key="ag_intake",
                            format_func=lambda c: f"{c}: {cases['intake'][c]['description'][:70]}…")
        st.markdown(f"> {md(cases['intake'][pick]['description'])}")
        exp = cases["intake"][pick]["expected"]
        rows = []
        for v in ("v1", "v2"):
            d = parse(agent_reply(v, pick)) or {}
            rows.append({f: d.get(f) for f in INTAKE_FIELDS})
        st.dataframe(pd.DataFrame([rows[0], rows[1], {f: exp.get(f) for f in INTAKE_FIELDS}],
                                  index=["v1", "v2", "Correct answer"]).astype(str).replace("None", "blank"),
                     width="stretch")
        st.caption("Blank means not stated; the right behaviour is to leave it blank and ask, not guess.")

    with st.expander("Memo agent: the CFO memo"):
        pick = st.selectbox("Business case", list(cases["memo"]), key="ag_memo")
        m1, m2 = st.columns(2)
        for c, v in ((m1, "v1"), (m2, "v2")):
            g = next(r for r in sc[v]["rows"]["memo"] if r["id"] == pick)
            c.markdown(f"**{v}** {'✅ every number traces to a fact' if g['faithful'] else '❌ invented: ' + ', '.join(f'{n:,.0f}' for n in g['invented'])}")
            c.markdown(md(agent_reply(v, pick)).replace("## ", "#### "))

    with st.expander("Red team agent: objections to a business case with a planted flaw"):
        pick = st.selectbox("Case", list(cases["redteam"]), key="ag_rt",
                            format_func=lambda c: "Sound case (nothing planted)" if not cases["redteam"][c]["planted"]
                            else f"Planted flaw: {AGENT_ASSUMPTIONS[cases['redteam'][c]['planted']]}"
                            + (" (held out)" if cases["redteam"][c].get("held_out") else ""))
        r1, r2 = st.columns(2)
        for c, v in ((r1, "v2"), (r2, "v3")):
            with c:
                st.markdown(f"**{v}**")
                show_objections(agent_reply(v, pick))

    st.divider()
    st.markdown("**Try the intake agent live**")
    if aws_ready():
        text = st.text_area("Describe an AI use case in a few sentences", key="ag_live",
                            placeholder="Our 5 buyers raise 400 purchase orders a month, 20 minutes each...")
        if st.button("Extract inputs", key="ag_live_go", disabled=not text.strip()):
            with st.spinner("Calling Amazon Nova Lite..."):
                try:
                    out = run_live("intake", {"description": text}, 3, profile=os.environ.get("AWS_PROFILE"))
                    d = json.loads(out["final"])
                    st.dataframe(pd.DataFrame([{f: d.get(f) for f in INTAKE_FIELDS}]).astype(str).replace("None", "blank"),
                                 width="stretch", hide_index=True)
                    missing = [f for f in INTAKE_FIELDS if d.get(f) is None]
                    if missing:
                        st.caption("Not stated, ask the manager: " + ", ".join(missing))
                    st.caption(f"{out['input_tokens']} + {out['output_tokens']} tokens, {out['latency_ms']} ms")
                except Exception as e:
                    st.error(f"Bedrock call failed: {e}")
    else:
        st.caption("Live mode runs only on a machine with AWS credentials (set AWS_PROFILE), so this public demo "
                   "shows saved results and cannot spend anyone's AWS credits.")


# ---------------------------------------------------------------- Shared results (so every page works on its own)
@st.cache_data(show_spinner="Training the model…")
def default_value() -> dict:
    """Prove-the-value result at the default plant costs and the most profitable threshold."""
    res = get_model_results()
    y, name, costs = res["y_test"], "Gradient boosting", Costs()
    proba = res["models"][name]["proba"]
    curve = sweep(y, proba, costs)
    thr = float(round(curve.loc[curve["annual_net"].idxmax(), "threshold"], 2))
    ranges = default_ranges(costs)
    top = sensitivity(y, proba, thr, costs, ranges).iloc[0]
    return {**evaluate(y, proba, thr, costs), "model": name, "band": bootstrap(y, proba, thr, costs),
            "cases": scenarios(y, proba, thr, costs, ranges), "costs": costs,
            "top_driver": top["assumption"].replace(" ($)", ""), "top_swing": top["swing"]}


def get_value() -> dict:
    return st.session_state.get("value") or default_value()


def get_readiness() -> dict:
    if "readiness" in st.session_state:
        return st.session_state["readiness"]
    dims = {d: sum(v) / len(v) for d, v in EXAMPLE_PILOT.items()}
    v, weakest = verdict(dims)
    return {"verdict": v, "avg": sum(dims.values()) / len(dims), "weakest": weakest}


def summarize_tracking(t: pd.DataFrame, gap: pd.DataFrame, source: str) -> dict:
    last = t.iloc[-1]
    delivered, promised = last["cum_actual"], last["cum_forecast"]
    worst = gap.loc[gap["effect"].idxmin()]
    return {
        "source": source, "months": len(t), "first": t["month"].iloc[0], "last": last["month"],
        "delivered": delivered, "promised": promised, "pct": delivered / promised if promised > 0 else np.nan,
        "adoption_last": last["adoption"], "low_adoption_months": t.loc[t["adoption_alert"], "month"].tolist(),
        "recall_now": t["recall_3m"].dropna().iloc[-1] if t["recall_3m"].notna().any() else np.nan,
        "drift_now": bool(last["drift_alert"]),
        "worst_cause": worst["cause"] if worst["effect"] < 0 else None, "worst_effect": worst["effect"],
    }


def default_tracking() -> dict:
    val, pipe = get_value(), load_pipeline()
    actuals, source = ((pipe["gold"][COLUMNS], "S3 → Databricks pipeline") if pipe
                       else (pd.read_csv("data/sample_actuals.csv"), "simulated sample"))
    t = track(actuals, val, val["costs"])
    return summarize_tracking(t, explain_gap(t, val, val["costs"]), source)


# ---------------------------------------------------------------- Layout helpers
SECTIONS = {1: "Plan the project", 2: "Plan the project", 3: "Plan the project", 4: "Plan the project",
            5: "After launch", 6: "Generative AI", 7: "Generative AI"}
ORDER = ["home", "pick", "value", "ready", "case", "track", "pilot", "agents"]


def page_header(step: int, title: str, intro: str):
    """Step label, question, one-paragraph context and an 'answer' box the page fills in once it has computed it."""
    st.markdown(f'<div class="eyebrow">Step {step} of 7 · {SECTIONS[step]}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(intro)
    box = st.container(key=f"answer_{step}")
    box.markdown('<div class="answer-label">The answer</div>', unsafe_allow_html=True)
    return box


def nav_footer(name: str):
    i = ORDER.index(name)
    st.divider()
    back, _, nxt = st.columns([2, 3, 2])
    if i > 0:
        back.page_link(PAGES[ORDER[i - 1]], label=f"← {PAGES[ORDER[i - 1]].title}")
    if i < len(ORDER) - 1:
        nxt.page_link(PAGES[ORDER[i + 1]], label=f"Next: {PAGES[ORDER[i + 1]].title} →")


# ---------------------------------------------------------------- Overview
TOUR = {
    "Plan the project": [("pick", "Which of 12 AI ideas to do first"),
                         ("value", "What it is worth in dollars, and how sure we are"),
                         ("ready", "Whether the team is ready to go live"),
                         ("case", "A one-page summary for finance")],
    "After launch": [("track", "Is it delivering? Forecast vs actual, adoption and drift, fed by an S3 → Databricks pipeline")],
    "Generative AI": [("pilot", "A measured Amazon Bedrock pilot: does an AI work-order writer pay for itself?"),
                      ("agents", "AI analyst agents, graded automatically on cases with known answers")],
}


def page_home():
    st.markdown('<div class="eyebrow">AI value engineering · portfolio project</div>', unsafe_allow_html=True)
    st.title("AI Use-Case ROI Navigator")
    st.markdown(
        "**Most AI projects never make it from pilot to production.** This app follows one project the whole way: "
        "pick the right idea, prove it in dollars, check the team is ready, keep checking the value after launch, "
        "and test where generative AI actually pays off. The example is predictive maintenance in a factory."
    )
    val, tk = get_value(), default_tracking()
    sc = load_agents()["scores"]["v3"]
    s2 = load_pilot()["summary"][2]
    k1, k2, k3 = st.columns(3)
    with k1.container(border=True, key="kpi_1"):
        st.metric("Value of predictive maintenance", f"{money(val['annual_net'])} a year")
        st.caption(md(f"Pays back in {months(val['payback_months'])}. Tuned for dollars, not accuracy."))
    with k2.container(border=True, key="kpi_2"):
        st.metric("Delivered after 12 months", f"{tk['pct']*100:.0f}% of forecast")
        st.caption("The gap is explained cause by cause, and a drift alert says when to retrain.")
    with k3.container(border=True, key="kpi_3"):
        st.metric("GenAI that you can trust", f"{s2['correct_on_real_failures']:.0%} right cause")
        st.caption(md(f"Once code checks the numbers and the AI only writes; agents caught "
                      f"{sc['redteam_caught']} of {sc['redteam_flaws']} planted flaws."))

    st.markdown("### Take the tour")
    cols = st.columns(3)
    for col, (section, steps) in zip(cols, TOUR.items()):
        with col.container(border=True, key=f"tour_{section.split()[0].lower()}"):
            st.markdown(f"**{section}**")
            for name, blurb in steps:
                st.page_link(PAGES[name], label=PAGES[name].title)
                st.caption(blurb)
    st.caption("Real: the AI4I 2020 machine dataset, the trained model, the Amazon Bedrock pilot and agent runs, the "
               "S3 and Databricks pipeline. Illustrative: plant costs, use-case scores and the 12-month rollout, "
               "all editable. Independent concept project.")
    nav_footer("home")


# ---------------------------------------------------------------- Style and navigation
st.markdown("""
<style>
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
[data-testid="stMainBlockContainer"] { max-width: 1150px; padding-top: 2.5rem; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * { animation: fadeUp 0.5s ease-out both; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(2) { animation-delay: 0.05s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(3) { animation-delay: 0.1s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(4) { animation-delay: 0.15s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(n+5) { animation-delay: 0.2s; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
.eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem; font-weight: 600; color: #0f6d74; }
.answer-label { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; font-weight: 700;
                color: #0f6d74; margin-bottom: -0.4rem; }
[class*="st-key-answer_"] { background: rgba(15, 109, 116, 0.08); border-left: 4px solid #0f6d74;
                            border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 1.08rem; }
[class*="st-key-kpi_"], [class*="st-key-tour_"] { transition: transform 0.2s ease, box-shadow 0.2s ease; }
[class*="st-key-kpi_"]:hover, [class*="st-key-tour_"]:hover { transform: translateY(-2px);
                                                              box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08); }
[data-testid="stPageLink"] a { border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 8px; padding: 0.3rem 0.8rem;
                               transition: background 0.2s ease, border-color 0.2s ease; }
[data-testid="stPageLink"] a:hover { border-color: #0f6d74; background: rgba(15, 109, 116, 0.08); }
.stAppDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

PAGES = {
    "home": st.Page(page_home, title="Overview", icon=":material/home:", default=True),
    "pick": st.Page(page_pick, title="1 · Pick a use case", url_path="pick"),
    "value": st.Page(page_value, title="2 · Prove the value", url_path="value"),
    "ready": st.Page(page_ready, title="3 · Check readiness", url_path="readiness"),
    "case": st.Page(page_case, title="4 · Business case", url_path="business-case"),
    "track": st.Page(page_track, title="5 · Track the value", url_path="track"),
    "pilot": st.Page(page_pilot, title="6 · GenAI pilot", url_path="genai-pilot"),
    "agents": st.Page(page_agents, title="7 · AI analyst agents", url_path="agents"),
}
st.navigation({"": [PAGES["home"]],
               "Plan the project": [PAGES[k] for k in ("pick", "value", "ready", "case")],
               "After launch": [PAGES["track"]],
               "Generative AI": [PAGES["pilot"], PAGES["agents"]]}, position="top").run()
