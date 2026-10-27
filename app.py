"""AI ROI Navigator: an AI business systems analysis case study

An independent project: a factory's downtime problem taken through to an AI investment decision and
a check of the value it delivers after launch.

Overview           - the lifecycle as one chart, my role, a closer look at each stage, conclusion
Before launch
  1 Evaluate AI solutions      - 12 AI options scored and matched to the business need, with reasons
  2 Build the business case    - a real predictive maintenance model, priced in dollars, with ranges and sensitivity
  3 Assess readiness           - readiness check, stakeholder impact, human review and escalation rules
  4 Prepare the decision       - recommendation, decision rule and a one-page business case
After launch
  5 Measure realized value     - forecast vs actual savings, reasons for the gap (fed by the S3 -> Databricks pipeline)
GenAI experiments (separate from the main case)
  A GenAI work orders          - measured Bedrock pilot of an AI work-order writer, and its ROI
  B GenAI business-case review - intake, memo and critic agents on Bedrock, graded on known answers

Every page follows one template: the question, the answer, three numbers, one chart and what to do next,
then 'Show the details' with the assumptions, method and data, then Back / Next links.
"""

import html
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring import (
    DEFAULT_FEAS_WEIGHTS,
    DEFAULT_VALUE_WEIGHTS,
    FEASIBILITY_CRITERIA,
    NEXT_STEP,
    THRESHOLD,
    VALUE_CRITERIA,
    score,
)
from src.brief import build_brief
from src.model import load_data, train_and_score
from src.matching import EXAMPLE_CLIENTS, INDUSTRIES, RISK_WEIGHT, URGENCY_WEIGHT, Client, match, problems, weights
from src.readiness import ACTIONS, DIMENSIONS, verdict
from src.genai import COST_LABELS, GenAIInputs
from src.genai import evaluate as genai_evaluate
from src.tracking import ADOPTION_FLOOR, COLUMNS, DRIFT_DROP, DRIFT_WINDOW, explain_gap, track, validate
from src.agents import ASSUMPTIONS as AGENT_ASSUMPTIONS, INTAKE_FIELDS, run_live
from src.agents import MODEL_ID as AGENT_MODEL_ID
from agents.grade import scorecard
from src.workorder import MODEL_ID, MODES, grade, parse, rule_alerts, summary as pilot_summary
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

st.set_page_config(page_title="AI Use-Case ROI Navigator", page_icon=":material/insights:", layout="wide")

QUADRANT_COLORS = {
    "Start now": "#1d7a45",
    "Fix a blocker first": "#0f6d74",
    "Small win": "#a25f00",
    "Not now": "#8a979c",
}


@st.cache_data
def manual_rule_alerts() -> np.ndarray:
    return rule_alerts()


def alternatives(r: dict, costs: Costs) -> dict:
    """The AI against two options without it: do nothing, and alerts from the maintenance manual's limits.

    Also the inspection cost above which the AI beats the rules on money (it raises far fewer false alarms)."""
    rules = evaluate(get_model_results()["y_test"], manual_rule_alerts(), 0.5, costs)
    extra_fa = rules["false_alarms"] - r["false_alarms"]
    gap = rules["annual_net"] - r["annual_net"]
    flip = costs.cost_false_alarm + gap / extra_fa if extra_fa > 0 and gap > 0 else None
    return {"rules": rules, "flip_inspection_cost": flip}


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



def bar_layout(fig, height=300, money_axis=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2, title=""),
                      xaxis_title="", yaxis_title="")
    if money_axis:
        fig.update_layout(**{money_axis: dict(tickformat="$,.0f", title="")})
    return fig


# ---------------------------------------------------------------- Page 1
MISTAKE_TOLERANCE = {"Low": "little tolerance for AI mistakes", "Medium": "some tolerance for AI mistakes",
                     "High": "high tolerance for AI mistakes"}
BLOCKER_FIX = {"data_readiness": "clean and connect the data before building",
               "ease_of_build": "start from a ready-made product rather than a custom build",
               "low_risk": "keep a person reviewing every output and start with one team",
               "speed_to_value": "agree an early milestone the client can see"}


def pick_client() -> Client:
    """Client chooser shown at the top of step 1: three example clients or your own."""
    options = list(EXAMPLE_CLIENTS) + ["Describe your own"]
    name = st.segmented_control("Business", options, default=options[0], key="client_pick") or options[0]
    if name != "Describe your own":
        c = EXAMPLE_CLIENTS[name]
        st.markdown(f'<div class="intro">{esc_html(c.blurb)}. Main problem: <b>{esc_html(c.problem.lower())}</b> · '
                    f'data quality {c.data_maturity}/5 · {esc_html(MISTAKE_TOLERANCE[c.risk_appetite])} · '
                    f'needs results {esc_html(c.urgency.lower())}.</div>', unsafe_allow_html=True)
        return c
    lib = load_use_cases()
    a, b, c3, d, e = st.columns([1.1, 1.5, 1, 1, 1.2])
    industry = a.selectbox("Industry", INDUSTRIES, key="cl_industry")
    problem = b.selectbox("Main problem", problems(lib, industry), key=f"cl_problem_{industry}")
    maturity = c3.slider("Data quality", 1, 5, 3, key="cl_data", help="1 = scattered spreadsheets, 5 = clean, connected data")
    risk = d.selectbox("Tolerance for AI mistakes", list(RISK_WEIGHT), index=1, key="cl_risk")
    urgency = e.selectbox("Needs results", list(URGENCY_WEIGHT), key="cl_urgency")
    return Client("Your business", industry, problem, maturity, risk, urgency)


def page_pick():
    answer, client, detail = page_header(1, "Which AI fits the business need?",
                                         "12 AI options, checked against what the business needs: its main problem, "
                                         "its data, how much risk it accepts and how fast it needs results.")
    with client:
        cl = pick_client()
    t_map, t_ranked, t_lib, t_how = detail.tabs(["Value vs effort map", "Full ranking",
                                                 "All 12 options", "How the match works"])
    with t_lib:
        st.caption("Each option scored 1–5 in general (5 is always better; for risk, 5 means low risk). "
                   "Edit any score and the match updates. Scores are illustrative.")
        library = st.data_editor(
            load_use_cases(), hide_index=True, width="stretch", key="library_edit",
            disabled=["use_case", "industry", "description", "serves", "solves"],
            column_config={col: st.column_config.NumberColumn(lbl, min_value=1, max_value=5, step=1)
                           for col, lbl in {**VALUE_CRITERIA, **FEASIBILITY_CRITERIA}.items()},
        )
    m = match(library, cl)
    m.insert(0, "rank", range(1, len(m) + 1))
    top = m.iloc[0]
    st.session_state["client"] = cl
    st.session_state["top_use_case"], st.session_state["top_quadrant"] = top["use_case"], top["quadrant"]
    fits = int(m["solves_problem"].sum())
    who = "this business" if cl.name == "Your business" else f"the {cl.name.lower()}"
    blocker = min(FEASIBILITY_CRITERIA, key=lambda k: top[k])
    next_txt = (f"Fix the blocker first: {FEASIBILITY_CRITERIA[blocker].lower()} is {top[blocker]:.0f}/5, so "
                f"{BLOCKER_FIX[blocker]}." if top["quadrant"] == "Fix a blocker first" else NEXT_STEP[top["quadrant"]])
    answer.markdown(md(
        f"**{top['use_case']}** for {who}. {top['description']}. "
        + ("It solves their main problem and is ready to start." if top["quadrant"] == "Start now" and top["solves_problem"]
           else "It solves their main problem, but one blocker needs fixing first." if top["solves_problem"]
           else f"None of the options targets {cl.problem.lower()} directly; this is the closest fit.")))

    with t_ranked:
        st.dataframe(m[["rank", "use_case", "solves", "value", "feasibility", "priority", "quadrant", "next_step"]],
                     hide_index=True, width="stretch")
    with t_map:
        plot_df = m.copy()
        dup = plot_df.groupby(["feasibility", "value"]).cumcount()
        plot_df["x_plot"] = plot_df["feasibility"] + dup * 0.07
        plot_df["y_plot"] = plot_df["value"] - dup * 0.07
        fig = px.scatter(
            plot_df, x="x_plot", y="y_plot", color="quadrant", color_discrete_map=QUADRANT_COLORS, text="rank",
            hover_name="use_case", hover_data={"x_plot": False, "y_plot": False, "rank": False, "solves": True,
                                               "value": True, "feasibility": True},
            range_x=[1, 5.2], range_y=[1, 5.2], labels={"x_plot": "Easier to do →", "y_plot": "More valuable →"},
            height=420,
        )
        fig.add_vline(x=THRESHOLD, line_dash="dot", line_color="#8a979c")
        fig.add_hline(y=THRESHOLD, line_dash="dot", line_color="#8a979c")
        fig.update_traces(textposition="middle center", marker=dict(size=24), textfont=dict(color="white", size=11))
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("Numbers on the dots are the rank. Top right: valuable and easy.")
    with t_how:
        vw, fw = weights(cl)
        st.markdown(
            "1. **Industry:** only options used in the business's industry count.\n"
            "2. **Main problem:** an option that solves it scores higher on value; one that doesn't scores lower.\n"
            "3. **Data quality:** good data makes an option easier to deliver, poor data harder.\n"
            f"4. **Tolerance for AI mistakes** ({cl.risk_appetite.lower()}): the lower it is, the more a risky option "
            f"is marked down ({fw['low_risk']:.0%} of the ease score).\n"
            f"5. **Urgency** ({cl.urgency.lower()}): quick results count for {fw['speed_to_value']:.0%} of the ease score.\n\n"
            "Fit score = value × ease, each 1–5, so 1–25. 'Start now' means both are 3.5 or more. "
            "Code: `src/matching.py`.")

    colors = [QUADRANT_COLORS[q] for q in m["quadrant"]]
    figb = go.Figure(go.Bar(y=m["use_case"], x=m["priority"], orientation="h", marker=dict(color=colors, cornerradius=6),
                            text=[f"{p:.1f}" + ("  · solves their problem" if s else "") for p, s in zip(m["priority"], m["solves_problem"])],
                            textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f} / 25<extra></extra>"))
    figb.update_layout(height=60 + 42 * len(m), margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(range=[0, 30], title="Fit score (out of 25)"),
                       yaxis=dict(autorange="reversed", title=""))

    with client:
        kpis([("Options that apply", f"{len(m)} of {len(library)}", f"used in {cl.industry.lower()}"),
              ("Solve the main problem", f"{fits}", cl.problem.lower()),
              ("Recommended", top["use_case"], top["quadrant"])])

        st.markdown(f"**How each solution fits {who}** &nbsp; " + " &nbsp; ".join(
            f'<span style="color:{c}">●</span> {q.lower()}' for q, c in QUADRANT_COLORS.items()), unsafe_allow_html=True)
        st.plotly_chart(figb, width="stretch")
        next_steps([f"**{w}**" if i == 0 else w for i, w in enumerate(top["why"])]
                   + [f"**Next:** {next_txt}"], label="Why it fits")
    nav_footer("pick")


# ---------------------------------------------------------------- Page 2
def page_value():
    answer, client, detail = page_header(2, "Is it worth it?",
                                         "Worked example: the factory, and the AI picked for it, which warns before a "
                                         "machine breaks down. Every breakdown it catches, misses or wrongly predicts "
                                         "is turned into dollars.")
    res = get_model_results()
    y_test = res["y_test"]
    t_costs, t_acc, t_risk, t_build = detail.tabs(["Costs", "Warning level", "How sure are we?", "Other options"])
    with t_costs:
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
        st.caption("Defaults are illustrative assumptions for a mid-size plant. Edit them to match your operation.")

    with t_acc:
        model_name = st.radio("Model", list(res["models"].keys()), index=1, horizontal=True, key="model_pick")
        proba = res["models"][model_name]["proba"]
        curve = sweep(y_test, proba, costs)
        best_money = curve.loc[curve["annual_net"].idxmax()]
        best_acc = curve.loc[curve["accuracy"].idxmax()]
        threshold = st.slider("Alert threshold: alert when failure probability is at least…",
                              0.01, 0.99, float(round(best_money["threshold"], 2)), 0.01, key="thr",
                              help="Lower = more sensitive: catches more failures but raises more false alarms.")
    r = evaluate(y_test, proba, threshold, costs)
    band = bootstrap(y_test, proba, threshold, costs)
    gap = best_money["annual_net"] - best_acc["annual_net"]

    with t_acc:
        if gap > 1:
            st.markdown(md(
                f"**Accuracy is not value.** The most accurate setting (threshold {best_acc['threshold']:.2f}, "
                f"{best_acc['accuracy']*100:.1f}% accurate) saves {money(best_acc['annual_net'])} a year; the most "
                f"profitable one (threshold {best_money['threshold']:.2f}) saves {money(best_money['annual_net'])}. "
                "A missed failure costs far more than a false alarm, so it pays to alert earlier."))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["annual_net"], name="Annual net savings ($)",
                                  line=dict(color="#0f6d74", width=3)))
        fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["accuracy"] * 100, name="Accuracy (%)",
                                  line=dict(color="#a25f00", dash="dot"), yaxis="y2"))
        fig2.add_vline(x=best_money["threshold"], line_color="#1d7a45", line_dash="dash",
                       annotation_text="most profitable", annotation_position="top left")
        fig2.add_vline(x=best_acc["threshold"], line_color="#a25f00", line_dash="dash",
                       annotation_text="most accurate", annotation_position="top right")
        fig2.add_vline(x=threshold, line_color="#8a979c", line_width=1, annotation_text="you", annotation_position="bottom right")
        fig2.update_layout(
            height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
            xaxis_title="Alert threshold", yaxis=dict(title="Annual net savings ($)", tickformat="$,.0f"),
            yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig2, width="stretch")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Failures caught / year", f"{r['caught']:.0f}")
        k1.caption(f"90% range: {band['caught'][0]:.0f} – {band['caught'][1]:.0f}")
        k2.metric("Failures missed / year", f"{r['missed']:.0f}")
        k2.caption(f"90% range: {band['missed'][0]:.0f} – {band['missed'][1]:.0f}")
        k3.metric("False alarms / year", f"{r['false_alarms']:.0f}")
        k3.caption(f"90% range: {band['false_alarms'][0]:.0f} – {band['false_alarms'][1]:.0f}")
        k4.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
        st.caption(md(
            f"{plain_language(r)} Ranges come from re-sampling the test set (bootstrap): it has only "
            f"{int(y_test.sum())} real failures, so the rates are estimates."))

    def fmt_input(key: str, v: float) -> str:
        return money(v) if ASSUMPTIONS[key].endswith("($)") else f"{v:,.0f}"

    base_vals = {k: float(getattr(costs, k)) for k in ASSUMPTIONS}
    defaults = default_ranges(costs)
    range_df = pd.DataFrame(
        {"Low": [defaults[k][0] for k in ASSUMPTIONS], "Base": list(base_vals.values()),
         "High": [defaults[k][1] for k in ASSUMPTIONS]},
        index=list(ASSUMPTIONS.values()),
    )
    with t_risk:
        st.markdown("Cost inputs are estimates. Set a realistic low and high for each one:")
        edited_ranges = st.data_editor(
            range_df, width="stretch", disabled=["Base"],
            # Reset to fresh defaults whenever the base inputs change
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
    low = cases["Low"]

    with t_risk:
        st.dataframe(pd.DataFrame({
            name: {"Annual net savings": money(c["annual_net"]), "3-year net value": money(c["three_year"]),
                   "3-year ROI": f"{c['roi_3y']*100:,.0f}%", "Payback": months(c["payback_months"]),
                   "First-year value": money(c["first_year"])}
            for name, c in cases.items()}), width="stretch")
        st.caption("Low case = every assumption at the end of its range that hurts value; high case = the end that helps.")
        labels = [f"{a.replace(' ($)', '')}  ({fmt_input(k, lo)} → {fmt_input(k, hi)})"
                  for a, k, lo, hi in zip(tornado["assumption"], tornado["key"], tornado["low_input"], tornado["high_input"])]
        figt = go.Figure()
        for side, name, color in [("low", "Input at low value", "#b4322c"), ("high", "Input at high value", "#0f6d74")]:
            figt.add_trace(go.Bar(
                y=labels, x=tornado[f"at_{side}"] - tornado["base"], base=tornado["base"], orientation="h", name=name,
                marker_color=color, customdata=tornado[f"at_{side}"],
                hovertemplate="%{y}<br>3-year net value: %{customdata:$,.0f}<extra>" + name + "</extra>",
            ))
        figt.add_vline(x=top["base"], line_color="#8a979c", line_width=1,
                       annotation_text=f"base {money(top['base'])}", annotation_position="top")
        figt.update_layout(
            barmode="overlay", height=60 + 45 * len(tornado), margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="3-year net value ($)", tickformat="$,.0f"),
            yaxis=dict(autorange="reversed", title=""), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(figt, width="stretch")
        st.caption("Each bar moves one assumption at a time; the longest bar is the biggest risk.")

    with t_build:
        rows = []
        for name, m in res["models"].items():
            cv = sweep(y_test, m["proba"], costs)
            b = cv.loc[cv["annual_net"].idxmax()]
            rows.append({"Model": name, "AUC": round(m["auc"], 3), "Best threshold": round(b["threshold"], 2),
                         "Recall": f"{b['recall']*100:.0f}%", "Precision": f"{b['precision']*100:.0f}%",
                         "Annual net savings": money(b["annual_net"]), "Payback (months)": round(b["payback_months"], 1)})
        st.markdown("**Models compared**, each at its most profitable threshold")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        alt = alternatives(r, costs)
        rl = alt["rules"]
        st.markdown("**Without AI: alerts from the maintenance manual's limits**")
        st.dataframe(pd.DataFrame([
            {"Option": "Do nothing", "Breakdowns caught": "0%", "Alerts that are real": "n/a",
             "False alarms a year": "0", "Annual net savings": money(0)},
            {"Option": "Rules from the manual", "Breakdowns caught": f"{rl['recall']*100:.0f}%",
             "Alerts that are real": f"{rl['precision']*100:.0f}%", "False alarms a year": f"{rl['false_alarms']:.0f}",
             "Annual net savings": money(rl["annual_net"])},
            {"Option": f"AI ({model_name.lower()})", "Breakdowns caught": f"{r['recall']*100:.0f}%",
             "Alerts that are real": f"{r['precision']*100:.0f}%", "False alarms a year": f"{r['false_alarms']:.0f}",
             "Annual net savings": money(r["annual_net"])},
        ]), hide_index=True, width="stretch")
        st.markdown(md(
            f"On this dataset the rules look as good as the AI, or better: they save {money(rl['annual_net'])} a year "
            f"against {money(r['annual_net'])}. That is because the dataset is synthetic and its breakdowns were "
            "generated from these same limits; in a real plant failures do not follow the manual this neatly. "
            f"The rules also raise **{rl['false_alarms']:.0f} false alarms a year against {r['false_alarms']:.0f}**: "
            f"only {rl['precision']*100:.0f}% of their alerts are real, and crew trust is already the weakest "
            "readiness area."
            + (f" The AI also wins on money once an inspection, including stopping the machine, costs more than "
               f"**{money(alt['flip_inspection_cost'])}** (assumed: {money(costs.cost_false_alarm)})."
               if alt["flip_inspection_cost"] else "")
            + " Recommendation: the AI, with the manual's limits kept as the rule check that explains each warning."))
        st.markdown("**Custom build vs ready-made solution**")
        cc1, cc2 = st.columns(2)
        custom_m = cc1.number_input("Months to deploy: custom build", 0, 24, 6, 1, key="custom_m")
        ready_m = cc2.number_input("Months to deploy: ready-made solution", 0, 24, 1, 1, key="ready_m")
        fy = lambda months: r["annual_net"] * max(0, 12 - months) / 12 - costs.deployment_cost
        st.markdown(md(f"First-year value: custom build **{money(fy(custom_m))}**, ready-made **{money(fy(ready_m))}**. "
                       f"Going live {custom_m - ready_m} months sooner is worth {money(fy(ready_m) - fy(custom_m))}."))
        st.caption(f"Trained on {res['n_train']:,} records, tested on {res['n_test']:,} it never saw "
                   f"({int(y_test.sum())} real failures). Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0). "
                   "Dollar results scale the test-set rates to your plant's failures per year.")

    st.session_state["value"] = {**r, "model": model_name, "band": band, "cases": cases, "costs": costs,
                                 "top_driver": top["assumption"].replace(" ($)", ""), "top_swing": top["swing"]}
    robust = low["annual_net"] > 0 and low["payback_months"] <= 36
    answer.markdown(md(
        f"**Yes: it saves {money(r['annual_net'])} a year and pays for itself in {months(r['payback_months'])}.** "
        + (f"Even if every cost estimate is too optimistic it still saves {money(low['annual_net'])} a year."
           if robust else f"But if the cost estimates are too optimistic it {'loses' if low['annual_net'] <= 0 else 'saves only'} "
           f"{money(abs(low['annual_net']))} a year, so firm them up first.")))

    figs = go.Figure(go.Bar(
        x=["Costs worse than expected", "As planned", "Costs better than expected"], y=[c["annual_net"] for c in cases.values()],
        text=[money(c["annual_net"]) for c in cases.values()], textposition="outside",
        marker_color=["#a25f00", "#0f6d74", "#1d7a45"]))
    bar_layout(figs, 280, "yaxis")
    figs.update_yaxes(range=[min(0, low["annual_net"]) * 1.2, max(c["annual_net"] for c in cases.values()) * 1.2])

    with client:
        kpis([("Projected savings a year", money(r["annual_net"]), f"likely {money_range(band['annual_net'])}"),
              ("Pays for itself in", months(r["payback_months"]), f"{money(costs.deployment_cost)} to set up"),
              ("Saved over 3 years", money(r["three_year"]), "after all AI costs")])
        be = break_even_cost(r, costs)
        with st.expander("How the savings are worked out", icon=":material/calculate:", expanded=True):
            st.markdown(md(
                "| Each year | Amount | How it is worked out |\n|---|---|---|\n"
                f"| Breakdowns today | {money(r['cost_without'])} | {costs.failures_per_year:.0f} breakdowns × "
                f"{money(costs.cost_missed)} downtime and emergency repair |\n"
                f"| Breakdowns the AI warns about in time | {money(r['caught'] * costs.cost_caught)} | "
                f"{r['caught']:.1f} become planned repairs × {money(costs.cost_caught)} |\n"
                f"| Breakdowns it misses | {money(r['missed'] * costs.cost_missed)} | {r['missed']:.1f} × "
                f"{money(costs.cost_missed)} |\n"
                f"| False alarms | {money(r['false_alarms'] * costs.cost_false_alarm)} | {r['false_alarms']:.1f} "
                f"inspections × {money(costs.cost_false_alarm)} |\n"
                f"| Running the AI | {money(costs.running_cost)} | licence, computing and support |\n"
                f"| Cost with the AI | {money(r['cost_with'])} | sum of the four lines above |\n"
                f"| **Net savings** | **{money(r['annual_net'])}** | today minus with the AI, after running costs |\n\n"
                "| One-off and over time | Amount | How it is worked out |\n|---|---|---|\n"
                f"| Setup and integration | {money(costs.deployment_cost)} | one-time, single estimate (not split "
                "into software, integration and training) |\n"
                f"| Payback | {months(r['payback_months'])} | {costs.months_to_deploy:.0f} months to go live, then "
                "savings cover the setup cost |\n"
                f"| 3-year net value | {money(r['three_year'])} | savings for {36 - costs.months_to_deploy:.0f} live "
                "months minus setup |\n"
                f"| 3-year ROI | {r['roi_3y']*100:,.0f}% | 3-year net value ÷ setup cost |\n"
                f"| Break-even | {money(be)} per breakdown | it stops paying only if a breakdown costs less than "
                f"this (assumed: {money(costs.cost_missed)}) |"))
            st.caption("Catch, miss and false-alarm rates are measured on 3,000 machine records the model never saw, "
                       "then scaled to this plant. Costs are assumptions: change them under Show the details, "
                       "where you also find the ranges and which assumption matters most.")
        st.markdown("**Savings a year, depending on how good our cost estimates turn out to be**")
        st.plotly_chart(figs, width="stretch")
        next_steps([
            md(f"**Agree the {top['assumption'].replace(' ($)', '').lower()} with finance.** It is the number that "
               f"changes the answer most ({money(top['swing'])} over 3 years)."),
            md("**Set the warning level to save the most money, not to be the most accurate**"
               + (f": that is worth {money(gap)} more a year." if gap > 1 else ".")),
            md(f"**Choose the AI over simpler rules.** Alerts from the maintenance manual save "
               f"{money(alt['rules']['annual_net'])} on this synthetic data but raise {alt['rules']['false_alarms']:.0f} "
               f"false alarms a year against {r['false_alarms']:.0f} (see Other options)."),
            md(f"**Go live soon.** Every month of delay costs about {money(r['annual_net'] / 12)}."),
        ])
    nav_footer("value")


# ---------------------------------------------------------------- Page 3
EXAMPLE_PILOT = {"Data": [4, 4], "Fits how people work": [2, 2], "Clear payoff": [4, 3],
                 "People trust it": [3, 2], "Clear ownership": [3, 3]}


STAKEHOLDER_IMPACT = [
    ("Maintenance planner", "Checks AI warnings and draft work orders instead of writing orders after a breakdown",
     "Will it bury us in false alarms?", "Warning level set to keep false alarms rare (step 2); planner approves every order"),
    ("Technicians", "Planned repairs at the next stop instead of emergency call-outs", "Will the warnings be right?",
     "Every warning gives its reason in plain words; outcomes are logged each month"),
    ("Operations director", "Owns the monthly value review", "Is it paying off?",
     "Step 5 compares savings with the forecast every month"),
    ("Finance", "Agrees the cost assumptions and signs off", "Are the savings real?",
     "Cost of a breakdown agreed first; savings measured from the plant's own records"),
    ("Data / IT lead", "Connects warnings to the maintenance system; sends the monthly file", "Who maintains it?",
     "Named owner; files that fail the checks are set aside automatically"),
]
OVERSIGHT_RULES = [
    "No work order is released without a planner's approval.",
    f"If the crew acts on fewer than {ADOPTION_FLOOR:.0%} of warnings in a month, the maintenance manager reviews the workflow.",
    f"If the AI catches {DRIFT_DROP*100:.0f}+ points fewer breakdowns than at launch over {DRIFT_WINDOW} months, "
    "the data / IT lead updates it with recent data.",
    "A warning for an unknown machine is held, not dropped, and IT is told the same day.",
    "A monthly file that fails the data checks is set aside with the reason and the plant is asked to resend it.",
]


def page_ready():
    answer, client, detail = page_header(3, "Is the team ready?",
                                         "Most AI projects stall because of people and process, not the technology. "
                                         "Ten statements in five areas, filled in for the example factory.")
    t_rate, t_people, t_rules = detail.tabs(["Rate your team", "People and process", "How the verdict works"])
    with t_people:
        st.markdown("**What changes for each group, and what worries them**")
        st.dataframe(pd.DataFrame(STAKEHOLDER_IMPACT, columns=["Who", "What changes", "Likely concern", "How it is handled"]),
                     hide_index=True, width="stretch")
        st.markdown("**Human review and escalation rules**")
        st.markdown("\n".join(f"- {r}" for r in OVERSIGHT_RULES))
        st.caption("Stakeholders are personas and the concerns are typical ones, not from interviews. The process maps "
                   "for today and with the AI are in docs/requirements.md.")
    dim_scores = {}
    with t_rate:
        st.caption("1 = not at all true, 5 = fully true.")
        cols = st.columns(2)
        for i, (dim, qs) in enumerate(DIMENSIONS.items()):
            with cols[i % 2]:
                st.markdown(f"**{dim}**")
                vals = [st.slider(q, 1, 5, EXAMPLE_PILOT[dim][j], key=f"rd_{dim}_{j}") for j, q in enumerate(qs)]
                dim_scores[dim] = sum(vals) / len(vals)
    with t_rules:
        st.markdown("- **Go**: average 3.5 or more and no area below 2.5\n"
                    "- **Fix first**: any area below 2.5, or average 2.5–3.5\n"
                    "- **Not ready**: average below 2.5")

    v, weakest = verdict(dim_scores)
    avg = sum(dim_scores.values()) / len(dim_scores)
    weak = " and ".join(f"**{d.lower()}**" for d, _ in weakest[:2])
    answer.markdown({"Go": f"**Yes, go.** Keep an eye on {weak}.",
                     "Fix first": f"**Not yet.** The model is fine; fix {weak} before launch.",
                     "Not ready": f"**No.** Too many gaps; start with {weak}."}[v])
    st.session_state["readiness"] = {"verdict": v, "avg": avg, "weakest": weakest, "dims": dim_scores}

    figr = px.bar(pd.DataFrame({"Area": list(dim_scores), "Score": list(dim_scores.values())}),
                  x="Score", y="Area", orientation="h", range_x=[0, 5], height=260, text="Score")
    figr.update_traces(marker_color=["#b4322c" if s < 2.5 else "#a25f00" if s < 3.5 else "#1d7a45"
                                     for s in dim_scores.values()], texttemplate="%{text:.1f}")
    figr.add_vline(x=2.5, line_dash="dot", line_color="#8a979c")
    figr.add_vline(x=3.5, line_dash="dot", line_color="#8a979c")
    figr.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="Score out of 5")

    with client:
        kpis([("Verdict", v, "Go · Fix first · Not ready"),
              ("Average score", f"{avg:.1f} / 5", "3.5 or more to go"),
              ("Weakest area", weakest[0][0], f"{weakest[0][1]:.1f} / 5")])
        st.plotly_chart(figr, width="stretch")
        next_steps([f"**{d}** ({s:.1f}): {ACTIONS[d]}" for d, s in weakest])
    nav_footer("ready")


# ---------------------------------------------------------------- Page 4
DECISION_RULES = [("Stop", "Expected savings are not positive"),
                  ("Revise", "Savings turn negative if costs are worse than expected, or the team is not ready"),
                  ("Proceed after fixes", "Savings stay positive in the worst case, but the team check says fix first"),
                  ("Proceed", "Savings stay positive in the worst case and the team is ready")]


def decision_rule(val: dict, rd: dict) -> tuple[str, str]:
    """The recommendation finance is asked to approve: (short label, one sentence)."""
    worst = val["cases"]["Low"]["annual_net"]
    if val["annual_net"] <= 0:
        return "Stop", "Even as planned it costs more than it saves."
    if worst <= 0:
        return "Revise", "Firm up the cost estimates first: it loses money if they are too optimistic."
    if rd["verdict"] == "Not ready":
        return "Revise", "The team is not ready: close the gaps and check again."
    if rd["verdict"] == "Fix first":
        fixes = " and ".join(f"“{d.lower()}”" for d, _ in rd["weakest"][:2])
        return "Proceed after fixes", (f"Close the readiness gaps first ({fixes}). It saves money even in the worst case "
                                       f"({money(worst)} a year).")
    return "Proceed", f"It saves money even in the worst case ({money(worst)} a year) and the team is ready."


GAP_OWNER = {"Data": "Data / IT lead", "Fits how people work": "Maintenance manager", "Clear payoff": "Finance",
             "People trust it": "Operations director", "Clear ownership": "Operations director"}
GAP_EFFECT = {"Data": "The AI misses breakdowns it should catch",
              "Fits how people work": "Warnings do not reach the crew; savings fall short",
              "Clear payoff": "Finance cannot verify the savings",
              "People trust it": "The crew ignores correct warnings",
              "Clear ownership": "Nobody notices when the AI gets weaker"}


def business_case_brief(val: dict, rd: dict) -> dict:
    """Everything the executive brief PDF shows, as formatted strings (layout is in src/brief.py)."""
    c, cases = val["costs"], val["cases"]
    rules = alternatives(val, c)["rules"]
    label, reason = decision_rule(val, rd)
    calc = [["Breakdowns today", money(val["cost_without"]), f"{c.failures_per_year:.0f} breakdowns x {money(c.cost_missed)}"],
            ["Warned in time, repaired as planned", money(val["caught"] * c.cost_caught),
             f"{val['caught']:.1f} x {money(c.cost_caught)} planned repair"],
            ["Missed breakdowns", money(val["missed"] * c.cost_missed), f"{val['missed']:.1f} x {money(c.cost_missed)}"],
            ["False alarms", money(val["false_alarms"] * c.cost_false_alarm),
             f"{val['false_alarms']:.1f} inspections x {money(c.cost_false_alarm)}"],
            ["Running the AI", money(c.running_cost), "licence, computing and support"],
            ["Net savings", money(val["annual_net"]), "breakdowns today minus the four costs above"]]
    risks = [[f"{d} ({s:.1f} / 5)", GAP_EFFECT[d], ACTIONS[d], GAP_OWNER[d]] for d, s in rd["weakest"][:2] if s < 3.5]
    risks += [[f"Cost of a breakdown is an estimate", f"Moves 3-year value by {money(val['top_swing'])}",
               "Agree it with finance from repair invoices before approval", "Finance"],
              ["The dataset is synthetic", "Accuracy in a real plant is likely lower; simple rules look unrealistically good",
               "Pilot on one line and measure catch rate and false alarms before scaling", "Data / IT lead"],
              ["The AI gets weaker over time", "Fewer breakdowns caught",
               f"Monthly check; update the AI if its catch rate drops {DRIFT_DROP*100:.0f}+ points over {DRIFT_WINDOW} months",
               "Data / IT lead"]]
    m = c.months_to_deploy
    return {
        "title": "Predictive maintenance for a mid-size factory", "client": "Mid-size factory, manufacturing",
        "author": "Chaitali Shekar", "decision": label, "decision_reason": reason,
        "kpis": [("Projected savings a year", money(val["annual_net"]), "after AI running costs"),
                 ("Pays for itself in", months(val["payback_months"]), f"{money(c.deployment_cost)} to set up"),
                 ("3-year ROI", f"{val['roi_3y']*100:,.0f}%", f"{money(val['three_year'])} net over 3 years"),
                 ("If costs are worse", money(cases["Low"]["annual_net"]), "savings a year, worst case")],
        "situation": (f"A mid-size parts maker has about {c.failures_per_year:.0f} unplanned machine breakdowns a year at "
                      f"about {money(c.cost_missed)} each in downtime and emergency repair: <b>{money(val['cost_without'])} "
                      "a year</b>. The machines already record temperature, speed, torque and tool wear, so a model can "
                      f"warn before a breakdown and turn it into a planned repair at about {money(c.cost_caught)}. "
                      "Predictive maintenance was the best fit of 12 AI options evaluated, and the only one aimed at "
                      "unplanned downtime."),
        "options": [["Do nothing", "0%", "0", money(0), f"Keeps losing {money(val['cost_without'])} a year"],
                    ["Alerts from the maintenance manual's limits", f"{rules['recall']*100:.0f}%",
                     f"{rules['false_alarms']:.0f}", money(rules["annual_net"]),
                     f"Only {rules['precision']*100:.0f}% of alerts are real. Strong here only because the synthetic "
                     "data was generated from these same limits"],
                    [f"AI warning model (selected)", f"{val['recall']*100:.0f}%", f"{val['false_alarms']:.0f}",
                     money(val["annual_net"]), f"{val['precision']*100:.0f}% of alerts are real, which protects crew trust; "
                     "the manual's limits are kept to explain each warning"]],
        "calc": calc,
        "cases": [[name, money(x["annual_net"]), money(x["three_year"]), months(x["payback_months"])]
                  for name, x in zip(["Costs worse than expected", "As planned", "Costs better than expected"],
                                     cases.values())],
        "driver": (f"Biggest driver: {val['top_driver'].lower()}, which moves 3-year value by {money(val['top_swing'])}. "
                   "Worse case = every cost assumption at the end of its range that hurts value."),
        "risks": risks,
        "rules": [list(r) for r in DECISION_RULES], "rule_index": [r[0] for r in DECISION_RULES].index(label),
        "plan": [["Month 0", f"Approve the {money(c.deployment_cost)} setup budget; agree the cost of a breakdown with finance",
                  "Savings stay positive in the worst case"],
                 [f"Months 1-{m:.0f}" if m > 1 else "Month 1",
                  "Connect warnings to the maintenance system, close the readiness gaps, train the crew",
                  "Readiness verdict is Go; all 13 acceptance tests pass"],
                 ["Launch", "Warnings go live on one line; a planner approves every work order",
                  "No work order released without approval"],
                 ["Every month", "Compare savings with the forecast and explain any gap",
                  f"Crew acts on {ADOPTION_FLOOR:.0%}+ of warnings; catch rate within {DRIFT_DROP*100:.0f} points of launch"],
                 ["Month 6", "Decide whether to scale to more lines, improve, or reconsider",
                  "At least 90% of the forecast delivered to scale"]],
        "evidence": [["Catch, miss and false-alarm rates", "Measured", "3,000 held-out machine records (AI4I 2020, synthetic)"],
                     ["Plant costs", "Assumption", "Illustrative defaults, editable in the app"],
                     ["AI option scores", "Estimate", "My scoring of 12 options on 7 criteria"],
                     ["Readiness scores", "Assessment", "Example ratings for the factory"],
                     ["Results after launch", "Simulated", "12 made-up months run through the S3 to Databricks pipeline"]],
    }


def page_case():
    answer, client, detail = page_header(4, "Ready for sign-off?",
                                         "Everything so far on one page for finance to approve. "
                                         "It updates when you change anything on the other pages.",
                                         detail_label="Preview the one-pager")
    val = get_value()
    rd = get_readiness()
    tk = st.session_state.get("tracking") or default_tracking()
    label, reason = decision_rule(val, rd)
    answer.markdown(md(f"**Recommendation: {label.lower()}.** {reason}"))
    rules_md = "| Recommendation | When |\n|---|---|\n" + "".join(f"| {a} | {b} |\n" for a, b in DECISION_RULES)
    tracking_md = ""
    if tk:
        pct_txt = f"{tk['pct']*100:.0f}% of forecast" if np.isfinite(tk["pct"]) else "forecast not positive"
        tracking_md = (
            f"\n## Value delivered after go-live ({tk['source']}, {tk['first']} to {tk['last']})\n"
            f"- Net savings to date: {money(tk['delivered'])} vs {money(tk['promised'])} forecast ({pct_txt})\n"
            + (f"- Biggest shortfall: {tk['worst_cause']} ({money(tk['worst_effect'])}). "
               f"{GAP_ACTIONS[tk['worst_cause']]}\n" if tk["worst_cause"] else "")
            + (f"- Crew acted on fewer than {ADOPTION_FLOOR:.0%} of warnings in: {', '.join(tk['low_adoption_months'])}\n"
               if tk["low_adoption_months"] else f"- Crew acted on at least {ADOPTION_FLOOR:.0%} of warnings every month\n")
            + ("- The AI is getting worse: it caught "
               f"{tk['recall_now']*100:.0f}% of breakdowns over the last {DRIFT_WINDOW} months, {DRIFT_DROP*100:.0f}+ points below launch. Update it with recent data.\n"
               if tk["drift_now"] else "- The AI is catching breakdowns as well as at launch\n")

        )
    summary = f"""# AI Business Case: Predictive Maintenance

**Client:** Mid-size factory (manufacturing). Main problem: unplanned downtime.
**Recommended solution:** Predictive maintenance, the best fit from a library of 12 AI solutions (step 1). Last client matched on step 1: {st.session_state['client'].name if 'client' in st.session_state else 'n/a'} → {st.session_state.get('top_use_case', 'n/a')} ({st.session_state.get('top_quadrant', 'n/a')}).

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
""" + "".join(f"- Fix: {d} ({s:.1f}): {ACTIONS[d]}\n" for d, s in rd.get("weakest", [])) + f"""
## Recommendation: {label}
{reason}

{rules_md}
Acceptance criteria: 12 user acceptance tests and a requirements traceability matrix (docs/requirements.md).
""" + tracking_md + """
## Key assumptions
- Plant costs as entered on step 2 (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
    with client:
        kpis([("Projected savings a year", money(val["annual_net"]), f"pays for itself in {months(val['payback_months'])}"),
              ("If costs are worse", money(val["cases"]["Low"]["annual_net"]), "savings a year, worst case"),
              ("Team ready?", rd["verdict"], f"average {rd['avg']:.1f} / 5")])
        st.markdown("**How the recommendation is decided**")
        st.markdown(rules_md.replace(f"| {label} |", f"| **{label}** (this case) |"))
        b1, b2, _ = st.columns([1.3, 1, 1.2])
        b1.download_button("Download the decision brief (PDF)", build_brief(business_case_brief(val, rd)),
                           file_name="ai_decision_brief.pdf", mime="application/pdf", key="dl_brief", type="primary",
                           icon=":material/picture_as_pdf:")
        b2.download_button("Plain-text version (.md)", summary, file_name="ai_business_case.md", key="dl_summary")
    with detail:
        st.markdown(md(summary))
    nav_footer("case")


# ---------------------------------------------------------------- Page 5
SHORTFALL = {"Failures (volume)": "the plant had fewer breakdowns to prevent than assumed",
             "Model recall": "the model caught fewer failures than in testing",
             "Flagged failures acted on": "the crew ignored some correct alerts",
             "False alarms": "more false alarms than expected"}
GAP_NEXT = {"Failures (volume)": "update the failures per year in the business case; this is not the model's fault.",
            "Model recall": "retrain the model on recent data.",
            "Flagged failures acted on": "fix the alert workflow and build the crew's trust.",
            "False alarms": "review the alert threshold."}
GAP_ACTIONS = {
    "Failures (volume)": "The plant had a different number of failures than the business case assumed. "
                         "This is not the model's fault; update failures per year on step 2.",
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


def monthly_vs_forecast(t: pd.DataFrame, ok_color: str, alert_color: str, height: int = 320) -> go.Figure:
    """Actual net savings per month as bars, the monthly forecast as a dashed line; alert months highlighted."""
    alert = t["adoption_alert"] | t["drift_alert"]
    why = np.where(t["drift_alert"] & t["adoption_alert"], "warnings ignored and AI getting worse",
                   np.where(t["drift_alert"], "AI getting worse", np.where(t["adoption_alert"], "warnings ignored", "on track")))
    fig = go.Figure([
        go.Bar(x=t["month"], y=t["actual"], name="Actual", showlegend=False, marker=dict(color=np.where(alert, alert_color, ok_color),
                                                                         cornerradius=6),
               customdata=why, hovertemplate="%{x}: %{y:$,.0f} (%{customdata})<extra></extra>"),
        go.Scatter(x=t["month"], y=t["forecast"], name="Promised", mode="lines", line=dict(color="#15222a", dash="dash", width=2),
                   hovertemplate="Promised %{y:$,.0f}<extra></extra>")])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(tickformat="$,.0f", title=""),
                      xaxis_title="", legend=dict(orientation="h", y=-0.2), bargap=0.25)
    return fig


def page_track():
    answer, client, detail = page_header(5, "Is it paying off?",
                                         "Every month after launch, the plant's repair records are turned into dollars "
                                         "and compared with what was promised.")
    val = get_value()
    sample = pd.read_csv("data/sample_actuals.csv")
    pipe = load_pipeline()
    t_src, t_health, t_gap, t_data = detail.tabs(["Where the data comes from", "Warnings", "Why the gap", "Monthly data"])
    with t_src:
        use_pipe = pipe is not None and st.radio(
            "Data source", ["Pipeline: S3 → Databricks gold table", "Bundled sample file"], horizontal=True,
            key="actuals_source").startswith("Pipeline")
        u1, u2 = st.columns([3, 1])
        upload = u1.file_uploader("Or upload your own monthly actuals (CSV, one row per month)", type="csv",
                                  key="actuals_upload")
        u2.download_button("Download template", sample.to_csv(index=False),
                           file_name="monthly_actuals_template.csv", key="dl_template")
    if upload is not None:
        actuals, source = pd.read_csv(upload), "uploaded file"
    elif use_pipe:
        actuals, source = pipe["gold"][COLUMNS], "S3 → Databricks pipeline"
    else:
        actuals, source = sample, "simulated sample"
    log = pipe["log"] if pipe and upload is None and use_pipe else None
    with t_src:
        if log:
            st.markdown(md(
                f"**How the data gets here:** each month the plant's file lands in secure cloud storage (Amazon S3), then "
                f"Databricks checks it and turns it into dollars: files as received → checked rows → final numbers. "
                f"Last update {log['refreshed_at'][:16].replace('T', ' ')} UTC: {log['bronze_rows']} files received, "
                f"{log['silver_rows']} passed the checks, {len(log['quarantined'])} set aside. "
                + ("The final numbers match this page exactly." if pipe["matches"] else "Warning: the final numbers differ from this page.")))
            for q in log["quarantined"]:
                st.markdown(md(f"- Quarantined `{q['file']}` ({q['month']}): {q['reason']}"))
        st.caption("The data is a simulated 12-month rollout, not real plant data: adoption is slow at first, then "
                   "the model starts to drift. Columns: month, alerts_raised, failures_flagged, failures_prevented, "
                   "failures_missed, false_alarms.")

    errors = validate(actuals)
    if errors:
        answer.error("This file can't be used:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.session_state.pop("tracking", None)
        nav_footer("track")
        return

    c = val["costs"]
    t = track(actuals, val, c)
    gap = explain_gap(t, val, c)
    last = t.iloc[-1]
    n = len(t)
    delivered, promised = last["cum_actual"], last["cum_forecast"]
    pct = delivered / promised if promised > 0 else np.nan
    recall_now = t["recall_3m"].dropna().iloc[-1] if t["recall_3m"].notna().any() else np.nan
    tk = st.session_state["tracking"] = summarize_tracking(t, gap, source)
    low_months = t.loc[t["adoption_alert"], "month"].tolist()
    drift_months = t.loc[t["drift_alert"], "month"].tolist()

    answer.markdown(md(
        f"**Partly: {pct*100:.0f}% of what was promised** after {n} months ({money(delivered)} of {money(promised)}). "
        + (f"The biggest reason: {SHORTFALL[tk['worst_cause']]}. " if tk["worst_cause"] else "")
        + ("And the AI is now catching fewer breakdowns." if tk["drift_now"] else "")))

    with t_health:
        st.markdown(
            (f"**Are people acting on warnings?** The crew acted on {last['adoption']*100:.0f}% of warnings last month "
             f"(target {ADOPTION_FLOOR:.0%})" + (f"; it was below target in {', '.join(low_months)}." if low_months else "."))
            + "  \n"
            + (f"**Is the AI getting worse?** Over the last {DRIFT_WINDOW} months it caught {recall_now*100:.0f}% of "
               f"breakdowns vs {val['recall']*100:.0f}% at launch; we warn when it is {DRIFT_DROP*100:.0f}+ points lower"
               + (f" (warned in {', '.join(drift_months)})." if drift_months else ".")))
        g1, g2 = st.columns(2)
        figa = go.Figure(go.Bar(x=t["month"], y=t["adoption"] * 100,
                                marker_color=["#b4322c" if a else "#1d7a45" for a in t["adoption_alert"]]))
        figa.add_hline(y=ADOPTION_FLOOR * 100, line_dash="dot", line_color="#8a979c",
                       annotation_text=f"{ADOPTION_FLOOR:.0%} target", annotation_position="top left")
        figa.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Alerts acted on (%)",
                           yaxis=dict(range=[0, 105], title=""), xaxis_title="")
        g1.plotly_chart(figa, width="stretch")
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=t["month"], y=t["recall_3m"] * 100, mode="lines+markers", name="Recall (3-month)",
                                  line=dict(color="#0f6d74")))
        figd.add_hline(y=val["recall"] * 100, line_color="#8a979c", annotation_text="launch", annotation_position="top left")
        figd.add_hline(y=(val["recall"] - DRIFT_DROP) * 100, line_dash="dot", line_color="#b4322c",
                       annotation_text="drift alert", annotation_position="bottom left")
        figd.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                           title=f"Failures caught, rolling {DRIFT_WINDOW} months (%)",
                           yaxis=dict(range=[0, 105], title=""), xaxis_title="")
        g2.plotly_chart(figd, width="stretch")

    with t_gap:
        figw = go.Figure(go.Waterfall(
            x=["Forecast"] + gap["cause"].tolist() + ["Actual"],
            measure=["absolute"] + ["relative"] * len(gap) + ["total"],
            y=[promised] + gap["effect"].tolist() + [0],
            text=[money(promised)] + [money(e) for e in gap["effect"]] + [money(delivered)], textposition="outside",
            increasing=dict(marker_color="#1d7a45"), decreasing=dict(marker_color="#b4322c"),
            totals=dict(marker_color="#0f6d74"), connector=dict(line=dict(color="#8a979c")),
        ))
        figw.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(tickformat="$,.0f", title=""),
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
        st.caption(md(
            "Each cause is swapped from its forecast value to its actual value one at a time, so the effects add up "
            "exactly to the gap. The forecast assumes every alert is acted on. Dollar costs per failure can't be "
            f"checked from counts; the biggest cost driver is **{val['top_driver']}**, so check it against real repair invoices."))

    with t_data:
        st.dataframe(t[COLUMNS + ["recall", "adoption", "actual", "forecast"]].round(2), hide_index=True, width="stretch")

    figc = monthly_vs_forecast(t, "#0f6d74", "#c98a2b", height=320)

    steps = []
    if tk["drift_now"]:
        steps.append(f"**Update the AI with recent data**: it now catches {recall_now*100:.0f}% of breakdowns, "
                     f"down from {val['recall']*100:.0f}% at launch.")
    if tk["worst_cause"] and not (tk["worst_cause"] == "Model recall" and tk["drift_now"]):
        cause = SHORTFALL[tk["worst_cause"]]
        steps.append(md(f"**{cause[0].upper() + cause[1:]}** ({money(tk['worst_effect'])}): {GAP_NEXT[tk['worst_cause']]}"))
    if last["adoption_alert"]:
        steps.append(f"**Win back the crew:** only {last['adoption']*100:.0f}% of warnings were acted on last month.")
    for q in (log or {}).get("quarantined", []):
        steps.append(f"**Ask the plant to resend the {q['month']} file**: it failed the data checks.")

    if not np.isfinite(pct) or delivered <= 0:
        call = ("Reconsider", "it has not saved money so far; check whether the problem and the forecast still hold.")
    elif pct >= 0.9:
        call = ("Scale", "it delivers close to the forecast; roll it out to more lines.")
    else:
        call = ("Improve, then scale", f"it pays, but delivered {pct*100:.0f}% of the forecast; fix the causes "
                                        "below and update the forecast before rolling it out further.")
    gap_lines = [f"**{SHORTFALL[cs][0].upper() + SHORTFALL[cs][1:]}**: {money(e)}"
                 for cs, e in zip(gap["cause"], gap["effect"]) if e < 0]
    steps = [md(f"**{call[0]}**: {call[1]}")] + steps

    with client:
        kpis([(f"Saved in {n} months", money(delivered), f"{money(delivered - promised)} vs forecast"),
              ("Of the year-1 target", f"{pct*100:.0f}%" if np.isfinite(pct) else "n/a", f"of {money(promised)}"),
              ("Breakdowns caught now", f"{recall_now*100:.0f}%" if np.isfinite(recall_now) else "n/a",
               f"{val['recall']*100:.0f}% at launch")])
        st.markdown("**Saved each month vs forecast** (amber = a month with a warning)")
        st.plotly_chart(figc, width="stretch")
        if gap_lines:
            st.markdown(md(f"**Why it fell short by {money(promised - delivered)}**  \n" + "  \n".join(gap_lines)))
        st.caption(f"Forecast: the {money(promised)} the business case promised for these {n} months. Delivered: the "
                   "same formula applied to the plant's monthly records. Simulated data for this example.")
        next_steps(steps, label="Scale, improve or reconsider?")
    nav_footer("track")


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
    answer, client, detail = page_header(6, "Can AI write the repair orders?",
                                         "When a breakdown warning comes in, an AI writing assistant (Amazon Bedrock) "
                                         "drafts the repair order. We tested it on 30 real warnings, two ways.")
    pilot = load_pilot()
    s1, s2 = pilot["summary"][1], pilot["summary"][2]
    t_res, t_roi, t_drafts = detail.tabs(["Test results", "Cost and payoff", "Compare drafts"])
    with t_roi:
        st.caption("Five cost categories from Dogra, \"How To Measure AI ROI In A Small Business\" (Forbes, 2026). "
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
    r2 = out[2]

    with t_roi:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net value, version 2", money(r2["net"]), f"{money(r2['net'] - out[1]['net'])} vs version 1")
        k2.metric("ROI, version 2", f"{r2['roi']*100:,.0f}%", f"{(r2['roi'] - out[1]['roi'])*100:+,.0f} pts vs version 1")
        k3.metric("Cash ROI, version 2", f"{r2['cash_roi']*100:,.0f}%", help="Counts only the share of saved hours that turns into cash.")
        k4.metric("Break-even review time", f"{r2['break_even_review_minutes']:.1f} min",
                  help="If reviewing a draft takes longer than this, the assistant loses money.")
        cost_rows = [{"Version": f"Version {v}", "Category": COST_LABELS[k], "Cost": c}
                     for v, r in out.items() for k, c in r["costs"].items()]
        figg = px.bar(pd.DataFrame(cost_rows), x="Cost", y="Version", color="Category", orientation="h", height=260,
                      color_discrete_sequence=["#0f6d74", "#8a979c", "#a25f00", "#1d7a45", "#b4322c"])
        for v, r in out.items():
            figg.add_trace(go.Scatter(x=[r["gain"]], y=[f"Version {v}"], mode="markers", name="Value of time saved" if v == 1 else None,
                                      showlegend=v == 1, marker=dict(symbol="line-ns", size=28, line=dict(width=3, color="#8a979c"))))
        figg.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(tickformat="$,.0f", title="Cost per year"),
                           yaxis_title="", legend=dict(orientation="h", y=-0.35), barmode="stack")
        st.plotly_chart(figg, width="stretch")
        st.markdown(md(
            f"**The model fee is not the cost.** Bedrock charges for {tasks:,} drafts come to **${r2['costs']['usage']:,.2f} a year**; "
            f"review time is **{money(r2['costs']['review'])}**. Version 1 costs almost the same to run but needs "
            f"{review[1]:.0f}-minute reviews, which "
            + ("wipes out the gain." if out[1]["net"] <= 0 else f"cuts net value to {money(out[1]['net'])}.")))
        st.caption("Tick = value of time saved. Training: 2 hours for 6 planners. Error rate = wrong-cause rate from "
                   "the pilot × share that slips past review.")

    with t_res:
        st.dataframe(pd.DataFrame({
            "Version 1: AI diagnoses and writes": [
                f"{s1['correct_mode_rate']:.0%}", f"{s1['correct_on_real_failures']:.0%}",
                f"{s1['false_alarm_handled_rate']:.0%}", f"${s1['cost_per_task']*1000:.3f}", f"{s1['median_latency_s']:.1f} s"],
            "Version 2: rule check diagnoses, AI writes": [
                f"{s2['correct_mode_rate']:.0%}", f"{s2['correct_on_real_failures']:.0%}",
                f"{s2['false_alarm_handled_rate']:.0%}", f"${s2['cost_per_task']*1000:.3f}", f"{s2['median_latency_s']:.1f} s"],
        }, index=["Right cause (all 30)", "Right cause (24 real failures)", "False alarm recognised (6)",
                  "Model cost per 1,000 drafts", "Median response time"]), width="stretch")
        st.markdown(
            "**Version 1 failed:** the model said \"overstrain\" 23 times, 17 of them when the readings were below the limit. "
            "It writes fluently but cannot be trusted to compare numbers. **Version 2** moves the diagnosis into a few lines of "
            "code and leaves the model to write. It kept the given diagnosis in all 30 drafts, but 4 still mention computer "
            "parts such as \"thermal grease on the CPU\", so a human review stays.")
        st.caption(f"Model {MODEL_ID}, temperature 0, max 500 output tokens, run on Bedrock in us-east-2. The 5 false alarms "
                   "version 2 still flags are machines with tools at 208–246 min of wear, where the manual says to replace "
                   "the tool anyway.")

    with t_drafts:
        cases = pilot["cases"]
        pick = st.selectbox("Alert", cases["case_id"], key="pilot_case",
                            format_func=lambda c: f"{c} (true cause: {MODES[cases.set_index('case_id').loc[c, 'true_mode']]})")
        d1, d2 = st.columns(2)
        for col, v in ((d1, 1), (d2, 2)):
            row = pilot["runs"][v].set_index("case_id").loc[pick]
            col.markdown(f"**Version {v}** · {'right cause' if row['correct_mode'] else 'wrong cause'}")
            with col:
                show_draft(row["reply"])

    answer.markdown(md(
        f"**Yes, if the AI only writes and simple rules do the diagnosis.** When the AI diagnosed too, it named the "
        f"right cause {s1['correct_on_real_failures']:.0%} of the time; with rules diagnosing, "
        f"{s2['correct_on_real_failures']:.0%}. The cost that matters is the time people spend checking drafts."))

    metrics = ["All 30 warnings", "24 real breakdowns", "6 false warnings"]
    figv = go.Figure([go.Bar(name=name, x=metrics, y=[s["correct_mode_rate"] * 100, s["correct_on_real_failures"] * 100,
                                                       s["false_alarm_handled_rate"] * 100],
                             text=[f"{x:.0%}" for x in (s["correct_mode_rate"], s["correct_on_real_failures"],
                                                         s["false_alarm_handled_rate"])],
                             textposition="outside", marker_color=color)
                      for name, s, color in (("Try 1: AI diagnoses and writes", s1, "#8a979c"),
                                             ("Try 2: rules diagnose, AI writes", s2, "#0f6d74"))])
    bar_layout(figv, 280)
    figv.update_yaxes(range=[0, 115], ticksuffix="%")

    with client:
        kpis([("Right diagnosis", f"{s1['correct_on_real_failures']:.0%} → {s2['correct_on_real_failures']:.0%}",
               "try 1 → try 2, real breakdowns"),
              ("AI cost", f"${s2['cost_per_task']*1000:.2f}", "per 1,000 drafts"),
              ("Saved each year", money(r2["net"]), "after all costs, try 2")])
        st.markdown("**Drafts that named the right cause**")
        st.plotly_chart(figv, width="stretch")
        next_steps([
            "**Let simple rules diagnose**, and the AI write the repair order.",
            "**Keep a person checking every draft**: 4 of 30 still mentioned computer parts.",
            f"**Time the checking** in a bigger test: above {r2['break_even_review_minutes']:.0f} minutes a draft, "
            "the assistant costs more than it saves.",
        ])
    nav_footer("pilot")


# ---------------------------------------------------------------- Page 7
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
        line = f"**{str(o.get('severity', 'n/a')).upper()}** · **{esc(o.get('assumption'))}**: {esc(o.get('issue') or o.get('why') or '')}"
        if o.get("issue") and o.get("why"):
            line += f"  \n{esc(o['why'])}"
        st.markdown(line)
        if o.get("test"):
            st.caption("30-day test: " + esc(o["test"]))
    if not d.get("objections"):
        st.caption("No objections.")
    st.markdown(f"Verdict: **{d.get('verdict', 'n/a')}**")


def page_agents():
    answer, client, detail = page_header(7, "Can AI do the analyst's paperwork?",
                                         "Three AI helpers: one turns a manager's description into numbers, one writes "
                                         "the memo for finance, one looks for weak spots like a skeptical CFO. "
                                         "Each was tested on cases with known answers.")
    AGENTS = load_agents()
    sc = AGENTS["scores"]
    cases = {k: {c["id"]: c for c in v} for k, v in AGENTS["cases"].items()}
    n_memo = len(cases["memo"])
    answer.markdown(md(
        f"**The routine parts, yes, once simple code checks the numbers and the AI only writes.** The number-taker went "
        f"from {sc['v1']['intake_field_accuracy']:.0%} to {sc['v3']['intake_field_accuracy']:.0%} right, and the critic "
        f"from {sc['v1']['redteam_caught']} to {sc['v3']['redteam_caught']} of {sc['v3']['redteam_flaws']} hidden mistakes "
        "caught. Deciding what to do about them stays with people."))

    versions = {"v1": "Try 1: AI does everything", "v2": "Try 2: code does the numbers", "v3": "Try 3: code checks, AI explains"}
    rows = {"Numbers taken down right": lambda s: s["intake_field_accuracy"],
            "Memos with no made-up numbers": lambda s: s["memo_faithful"] / n_memo,
            "Hidden mistakes caught": lambda s: s["redteam_caught"] / s["redteam_flaws"]}
    figv = go.Figure([go.Bar(name=label, x=list(rows), y=[f(sc[v]) * 100 for f in rows.values()],
                             text=[f"{f(sc[v]):.0%}" for f in rows.values()], textposition="outside", marker_color=color)
                      for (v, label), color in zip(versions.items(), ("#c4ccd0", "#8a979c", "#0f6d74"))])
    bar_layout(figv, 280)
    figv.update_yaxes(range=[0, 115], ticksuffix="%")

    with client:
        kpis([("Numbers taken down right", f"{sc['v1']['intake_field_accuracy']:.0%} → {sc['v3']['intake_field_accuracy']:.0%}",
               "try 1 → try 3"),
              ("Hidden mistakes caught", f"{sc['v3']['redteam_caught']} of {sc['v3']['redteam_flaws']}",
               f"{sc['v3']['redteam_clean_high']} false alarms"),
              ("Cost of the whole test", f"${sc['v3']['total_cost']:.4f}", "less than a cent")])
        st.markdown("**Score on each try**")
        st.plotly_chart(figv, width="stretch")
        next_steps([
            "**Let code do the maths and checks, and the AI do the writing.** Every improvement came from that split.",
            "**Test AI on cases with known answers before trusting it.** Try 1 read well and was mostly wrong.",
            "**Keep a person on the decision.** The test cases were written knowing the mistakes, so this is not a "
            "blind test.",
        ], label="What this means for a team")

    t_sc, t_in, t_memo, t_rt, t_live = detail.tabs(["Scorecard", "Number-taker", "Memo writer", "Critic", "Try it live"])

    with t_sc:
        col = lambda v: [
            f"{sc[v]['intake_field_accuracy']:.0%}", f"{sc[v]['intake_invented_fields']}",
            f"{sc[v]['memo_faithful']} of {n_memo}", f"{sc[v]['redteam_caught']} of {sc[v]['redteam_flaws']}",
            f"{sc[v]['redteam_caught_first']} of {sc[v]['redteam_flaws']}", f"{sc[v]['redteam_clean_high']}",
            f"${sc[v]['total_cost']:.4f}"]
        st.dataframe(pd.DataFrame({label: col(v) for v, label in versions.items()},
                                  index=["Number-taker: numbers right (8 descriptions)", "Number-taker: numbers made up when not given",
                                         "Memo writer: memos with no made-up numbers", "Critic: hidden mistake flagged as serious",
                                         "Critic: hidden mistake listed first", "Critic: false 'serious' flags on the sound case",
                                         "Cost of the whole eval run"]), width="stretch")
        st.markdown(md(
            "**What changed.** v1 wrote arithmetic like `300 * 52` inside its JSON, invented a $5,000 figure in a memo, and gave "
            "nearly the same five objections to every case. **v2:** the model only quotes the numbers it was given; code "
            "converts weeks to years, checks each quote is really in the text, and fills memo placeholders like "
            "`{First-year ROI}`. The number-taker and memo writer became exact, but the critic still called \"15 of 20 minutes saved\" "
            "impossible in every case. **v3:** a code checklist with one plausibility rule per assumption decides what is "
            "wrong and how serious it is; the model writes the CFO's explanation and a 30-day test."))
        st.caption(
            f"Model {AGENT_MODEL_ID}, temperature 0, us-east-2. Honest limits: the two held-out flaws (hourly cost $400, zero "
            f"training) were caught, but they were known when the checklist was written. In v3 the model explained "
            f"{sc['v3']['redteam_explained']} of {sc['v3']['redteam_flags']} flags; once it used the wrong assumption name, "
            "and code showed the flag without an explanation instead of guessing.")

    with t_in:
        pick = st.selectbox("Description", list(cases["intake"]), key="ag_intake",
                            format_func=lambda c: f"{c}: {cases['intake'][c]['description'][:70]}…")
        st.markdown(f"> {md(cases['intake'][pick]['description'])}")
        exp = cases["intake"][pick]["expected"]
        rows = [{f: (parse(agent_reply(v, pick)) or {}).get(f) for f in INTAKE_FIELDS} for v in ("v1", "v2")]
        st.dataframe(pd.DataFrame([rows[0], rows[1], {f: exp.get(f) for f in INTAKE_FIELDS}],
                                  index=["v1", "v2", "Correct answer"]).astype(str).replace("None", "blank"),
                     width="stretch")
        st.caption("Blank means not stated; the right behaviour is to leave it blank and ask, not guess.")

    with t_memo:
        pick = st.selectbox("Business case", list(cases["memo"]), key="ag_memo")
        m1, m2 = st.columns(2)
        for c, v in ((m1, "v1"), (m2, "v2")):
            g = next(r for r in sc[v]["rows"]["memo"] if r["id"] == pick)
            c.markdown(f"**{v}** {'· every number traces to a fact' if g['faithful'] else '· invented: ' + ', '.join(f'{n:,.0f}' for n in g['invented'])}")
            c.markdown(md(agent_reply(v, pick)).replace("## ", "#### "))

    with t_rt:
        pick = st.selectbox("Case", list(cases["redteam"]), key="ag_rt",
                            format_func=lambda c: "Sound case (nothing planted)" if not cases["redteam"][c]["planted"]
                            else f"Planted flaw: {AGENT_ASSUMPTIONS[cases['redteam'][c]['planted']]}"
                            + (" (held out)" if cases["redteam"][c].get("held_out") else ""))
        r1, r2 = st.columns(2)
        for c, v in ((r1, "v2"), (r2, "v3")):
            with c:
                st.markdown(f"**{v}**")
                show_objections(agent_reply(v, pick))

    with t_live:
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
    nav_footer("agents")

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
SECTIONS = {1: "Before launch", 2: "Before launch", 3: "Before launch", 4: "Before launch", 5: "After launch"}
EXPERIMENTS = {6: "A", 7: "B"}   # the GenAI experiments sit outside the numbered lifecycle
ORDER = ["home", "pick", "value", "ready", "case", "track", "pilot", "agents"]


def esc_html(text: str) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def page_header(step: int, title: str, intro: str, detail_label: str = "Show the details"):
    """Every page: question, one line of context, the answer, the client view, then the analysis on demand.

    Returns (answer, client, detail) containers; the page fills them in any order, so inputs inside the
    detail tabs can be read before the answer and the client view are written."""
    eyebrow = (f"GenAI experiment {EXPERIMENTS[step]} · separate from the main case" if step in EXPERIMENTS
               else f"Stage {step} of 5 · {SECTIONS[step]}")
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    intro = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", intro)
    st.markdown(f'<div class="intro">{intro}</div>', unsafe_allow_html=True)
    answer = st.container(key=f"answer_{step}")
    answer.markdown('<div class="box-label">The answer</div>', unsafe_allow_html=True)
    client = st.container()
    detail = st.expander(detail_label, icon=":material/query_stats:")
    return answer, client, detail


def kpis(items: list[tuple[str, str, str]]):
    """Three headline numbers: (label, value, note)."""
    for i, (col, (label, value, note)) in enumerate(zip(st.columns(len(items)), items)):
        with col.container(border=True, key=f"kpi_{i}"):
            st.markdown(f'<div class="kpi-label">{esc_html(label)}</div><div class="kpi-value">{esc_html(value)}</div>'
                        f'<div class="kpi-note">{esc_html(note)}</div>', unsafe_allow_html=True)


def next_steps(items: list[str], label: str = "What to do next"):
    with st.container(key=f"next_{re.sub(r'[^a-z]', '', label.split()[0].lower())}"):
        st.markdown(f'<div class="box-label">{label}</div>', unsafe_allow_html=True)
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(items, 1)))


def nav_footer(name: str):
    i = ORDER.index(name)
    st.divider()
    back, _, nxt = st.columns([2, 3, 2])
    if i > 0:
        back.page_link(PAGES[ORDER[i - 1]], label=f"← {PAGES[ORDER[i - 1]].title}")
    if i < len(ORDER) - 1:
        nxt.page_link(PAGES[ORDER[i + 1]], label=f"Next: {PAGES[ORDER[i + 1]].title} →")


# ---------------------------------------------------------------- Overview
SAGE, LIME, INK = "#c8d2c3", "#d4ee5e", "#1d2521"
REPO = "https://github.com/chaialishekar/AI-ROI-Navigator/blob/main"
DOCS = {"Requirements pack": f"{REPO}/docs/requirements.md",
        "Process maps (as-is and to-be)": f"{REPO}/docs/requirements.md#2-process-map-from-warning-to-repair-today-and-with-ai",
        "Acceptance tests and traceability": f"{REPO}/docs/requirements.md#5-user-acceptance-tests",
        "User stories and stakeholders": f"{REPO}/docs/solution_design.md",
        "Roadmap": f"{REPO}/docs/roadmap.md"}


@st.cache_data
def accuracy_vs_profit() -> dict:
    """Savings at the most accurate vs the most profitable alert threshold, default plant costs."""
    res = get_model_results()
    curve = sweep(res["y_test"], res["models"]["Gradient boosting"]["proba"], Costs())
    return {"accurate": float(curve.loc[curve["accuracy"].idxmax(), "annual_net"]),
            "profitable": float(curve.loc[curve["annual_net"].idxmax(), "annual_net"])}


@st.cache_data
def failure_causes() -> dict:
    """Breakdowns by cause in the 10,000 machine records (a breakdown can have more than one cause)."""
    df = load_data()
    names = {"TWF": "Tool wear", "HDF": "Heat", "PWF": "Power", "OSF": "Overstrain", "RNF": "Random"}
    return {names[k]: int(df[k].sum()) for k in names}


def default_track_frame() -> pd.DataFrame:
    val, pipe = get_value(), load_pipeline()
    actuals = pipe["gold"][COLUMNS] if pipe else pd.read_csv("data/sample_actuals.csv")
    return track(actuals, val, val["costs"])


def mini_chart(fig: go.Figure, height: int = 260) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
    fig.update_xaxes(showgrid=False, zeroline=False, title="")
    fig.update_yaxes(showgrid=False, zeroline=False, title="", showticklabels=False)
    return fig


def mini_bars(labels, values, texts, highlight, horizontal=False) -> go.Figure:
    colors = [LIME if h else SAGE for h in highlight]
    bar = (go.Bar(y=labels, x=values, orientation="h") if horizontal else go.Bar(x=labels, y=values))
    bar.update(text=texts, textposition="outside", cliponaxis=False, marker=dict(color=colors, cornerradius=14),
               hoverinfo="skip", textfont=dict(size=15, color=INK))
    fig = mini_chart(go.Figure(bar))
    if horizontal:
        fig.update_yaxes(showticklabels=True, autorange="reversed")
        fig.update_xaxes(showticklabels=False, range=[0, max(values) * 1.2])
    else:
        fig.update_yaxes(range=[0, max(values) * 1.25])
    return fig


def break_even_cost(val: dict, c: Costs) -> float:
    """Cost of a breakdown at which the AI stops paying: savings = caught × (breakdown − planned repair) − other costs."""
    if val["caught"] <= 0:
        return float("inf")
    return c.cost_caught + (val["false_alarms"] * c.cost_false_alarm + c.running_cost) / val["caught"]


STAGES = ["0 · Problem", "1 · Solutions", "2 · Business case", "3 · Readiness", "4 · Decision", "5 · Results",
          "A · Work orders", "B · Case review"]


def tour_content(stage: str) -> dict:
    """For one stage: the business question, how it was analysed, what was produced, the decision it supported,
    what kind of evidence the headline number is, and a small chart."""
    val, rd = get_value(), get_readiness()
    c = val["costs"]
    if stage.startswith("0"):
        causes = failure_causes()
        return dict(page=None, q="How big is the problem, and what would success look like?",
                    how=f"Set the baseline: {c.failures_per_year:.0f} surprise breakdowns a year at "
                        f"{money(c.cost_missed)} each in downtime and emergency repair. Broke the "
                        f"{int(load_data()['Machine failure'].sum())} breakdowns in the machine data down by cause, mapped how repairs "
                        "happen today, and listed who is involved: operations, maintenance, finance and IT.",
                    made="Problem statement and baseline, stakeholder list, as-is process map, user stories with "
                         "success measures.",
                    docs=["Process maps (as-is and to-be)", "User stories and stakeholders"],
                    decision=f"Worth solving: {money(c.failures_per_year * c.cost_missed)} a year is at stake.",
                    evidence="Baseline cost is an assumption you can edit; the breakdown data is real.",
                    chart_title="Breakdown causes in 10,000 real machine records",
                    fig=mini_bars(list(causes), list(causes.values()), [str(v) for v in causes.values()],
                                  [False] * len(causes), horizontal=True))
    if stage.startswith("1"):
        m = match(load_use_cases(), EXAMPLE_CLIENTS["Mid-size factory"])
        return dict(page="pick", q="Which AI fits this need, and why that one?",
                    how="Scored 12 AI options on 7 criteria (3 for value, 4 for ease, including how ready the data is), "
                        "then adjusted the scores for this factory's main problem, data quality, tolerance for AI "
                        "mistakes and how fast it needs results. In stage 2 the pick is also priced against doing "
                        "nothing and against simple alerts from the maintenance manual.",
                    made="Weighted decision matrix, value vs effort map, and the reasons for the pick.",
                    docs=[],
                    decision=f"{m.iloc[0]['use_case']}: the only option that solves unplanned downtime, and ready to start.",
                    evidence="Option scores are my estimates; the method is shown and the weights can be changed.",
                    chart_title="Fit score for the factory (green = solves its main problem)",
                    fig=mini_bars(m["use_case"], m["priority"], [f"{p:.1f}" for p in m["priority"]],
                                  m["solves_problem"], horizontal=True))
    if stage.startswith("2"):
        ap = accuracy_vs_profit()
        low = val["cases"]["Low"]["annual_net"]
        rules = alternatives(val, c)["rules"]
        return dict(page="value", q="Does it save more than it costs, and how sure are we?",
                    how="Trained a breakdown-warning model on real machine data and turned every catch, miss and false "
                        "alarm into dollars. Set the warning level to save the most money rather than to be the most "
                        f"accurate (worth {money(ap['profitable'] - ap['accurate'])} a year more), then tested best "
                        "and worst cases and which assumption matters most.",
                    made="Financial model: savings, costs, payback, 3-year ROI, ranges and sensitivity analysis.",
                    docs=[],
                    decision=f"Invest: {money(val['annual_net'])} a year after running costs, {money(c.deployment_cost)} "
                             f"to set up, pays back in {months(val['payback_months'])}. Still {money(low)} a year if "
                             f"every cost estimate is too optimistic. Rules from the manual save {money(rules['annual_net'])} "
                             f"on this synthetic data but raise {rules['false_alarms']:.0f} false alarms a year against "
                             f"{val['false_alarms']:.0f}, so the AI is recommended.",
                    evidence="Projection: real model results, assumed plant costs.",
                    chart_title="Cost of breakdowns each year",
                    fig=mini_bars(["Today", "With the AI", "Saved"],
                                  [val["cost_without"], val["cost_with"], val["annual_net"]],
                                  [money(val["cost_without"]), money(val["cost_with"]), money(val["annual_net"])],
                                  [False, False, True]))
    if stage.startswith("3"):
        dims = rd.get("dims") or {d: sum(v) / len(v) for d, v in EXAMPLE_PILOT.items()}
        weak = [d for d, _ in rd["weakest"][:2]]
        return dict(page="ready", q="Will people use it, and what has to change first?",
                    how="Rated the team on 10 statements in 5 areas, mapped how the repair process changes with the AI, "
                        "and listed what changes for each group and what worries them.",
                    made="Readiness score with fixes, to-be process map, stakeholder impact table, human review and "
                         "escalation rules.",
                    docs=["Process maps (as-is and to-be)"],
                    decision=(f"Close two gaps before launch: “{weak[0].lower()}” ({rd['weakest'][0][1]:.1f} / 5) and "
                              f"“{weak[1].lower()}” ({rd['weakest'][1][1]:.1f} / 5)." if rd["verdict"] != "Go"
                              else "Ready to launch."),
                    evidence="Assessment: example ratings for the factory.",
                    chart_title="Score out of 5 (green = fix before launch)",
                    fig=mini_bars(list(dims), list(dims.values()), [f"{v:.1f}" for v in dims.values()],
                                  [d in weak for d in dims], horizontal=True))
    if stage.startswith("4"):
        cases = val["cases"]
        return dict(page="case", q="What exactly is finance being asked to approve?",
                    how="Put the value, the range of outcomes, the risks and the readiness gaps on one page, with a "
                        "rule for when to proceed, fix first, revise or stop.",
                    made="One-page business case (download), decision rule, 12 acceptance tests, and a traceability "
                         "matrix linking each requirement to its test.",
                    docs=["Acceptance tests and traceability"],
                    decision=": ".join(decision_rule(val, rd)),
                    evidence="Recommendation, not an approval: no real finance team signed it.",
                    chart_title="Savings a year if costs turn out worse, as planned, or better",
                    fig=mini_bars(["Worse", "As planned", "Better"], [x["annual_net"] for x in cases.values()],
                                  [money(x["annual_net"]) for x in cases.values()], [False, True, False]))
    if stage.startswith("5"):
        t = default_track_frame()
        tk, gap = default_tracking(), explain_gap(t, val, c)
        gap = gap[gap["effect"] < 0].sort_values("effect")
        return dict(page="track", q="After 12 months, did it deliver what was promised?",
                    how="Turned each month's repair records into dollars with the same formula as the forecast, then "
                        "split the gap into its causes, one at a time, so the parts add up exactly.",
                    made="Forecast-vs-actual tracking, warnings when the crew ignores alerts or the AI gets weaker, and "
                         "the reasons for the gap.",
                    docs=[],
                    decision=f"Improve, then scale. It delivered {money(tk['delivered'])} of the {money(tk['promised'])} "
                             f"promised ({tk['pct']*100:.0f}%). The plant had fewer breakdowns than assumed (update the "
                             f"forecast), the crew ignored warnings in {len(tk['low_adoption_months'])} months (fix the "
                             "workflow) and " + ("the AI got weaker late in the year (update it with recent data)."
                                                 if tk["drift_now"] else "the AI caught fewer breakdowns than in testing."),
                    evidence="Simulated: 12 made-up months, run through the real data pipeline.",
                    chart_title="Why it fell short of the promise",
                    fig=mini_bars([SHORTFALL[x].split(" than")[0].replace("the ", "").capitalize() for x in gap["cause"]],
                                  list(-gap["effect"]), [money(e) for e in gap["effect"]],
                                  [False] * len(gap), horizontal=True))
    if stage.startswith("A"):
        pilot = load_pilot()
        s1, s2 = pilot["summary"][1], pilot["summary"][2]
        g2 = pilot["runs"][2]
        real = int((g2["true_mode"] != "NONE").sum())
        fa = len(g2) - real
        right1, right2 = round(s1["correct_on_real_failures"] * real), round(s2["correct_on_real_failures"] * real)
        return dict(page="pilot", q="Can GenAI draft repair orders a planner can trust?",
                    how=f"Sent {len(g2)} warnings from the test data ({real} real breakdowns, {fa} false alarms) to "
                        "Amazon Nova Micro on Bedrock and graded each draft against the true cause, in two versions.",
                    made="Graded results for both versions, cost per draft, and payback including review time.",
                    docs=[],
                    decision=f"Use GenAI to write, not to diagnose. Right cause on real breakdowns: {right1} of {real} "
                             f"when it diagnosed, {right2} of {real} when rules diagnosed and it only wrote. On false "
                             f"alarms it still named a fault in {fa - round(s2['false_alarm_handled_rate'] * fa)} of "
                             f"{fa}, so a planner reviews every draft.",
                    evidence=f"Measured on {len(g2)} cases: a small sample, not proof of reliability.",
                    chart_title=f"Right cause named, out of {real} real breakdowns",
                    fig=mini_bars(["AI diagnoses", "Rules diagnose, AI writes"], [right1, right2],
                                  [f"{right1} of {real}", f"{right2} of {real}"], [False, True]))
    sc = load_agents()["scores"]
    caught = [sc[v]["redteam_caught"] for v in ("v1", "v2", "v3")]
    n = sc["v3"]["redteam_flaws"]
    return dict(page="agents", q="Can GenAI spot errors in a business case?",
                how="Built three AI helpers on Amazon Bedrock (take down the numbers, write the memo, critique the "
                    "case) and tested them on 19 cases with known answers, including business cases with errors I seeded.",
                made="Scorecard for each version and the grading code.",
                docs=[],
                decision=f"Useful as a first check, not a replacement for a finance review. The critic went from "
                         f"{caught[0]} to {caught[2]} of {n} seeded errors once code checked the numbers, with "
                         f"{sc['v3']['redteam_clean_high']} false alarms on the sound case.",
                evidence="Measured on test cases I wrote: not a blind test.",
                chart_title=f"Seeded errors caught, out of {n}",
                fig=mini_bars(["Version 1", "Version 2", "Version 3"], caught, [str(x) for x in caught],
                              [False, False, True]))


def flow_node(page: str | None, num: str, title: str, desc: str, result: str, kind: str) -> str:
    """One stage card. kind says what the result is: a baseline, a decision, a projection, a result."""
    inner = (f'<span class="node-top"><span class="num">{num}</span>'
             + ('<span class="go">→</span>' if page else "") + '</span>'
             f'<span class="node-title">{esc_html(title)}</span><span class="node-q">{esc_html(desc)}</span>'
             f'<span class="node-kind">{esc_html(kind)}</span><span class="node-res">{esc_html(result)}</span>')
    if page is None:
        return f'<div class="node need">{inner}</div>'
    return f'<a class="node" href="{PAGES[page].url_path}" target="_self" aria-label="{esc_html(title)}">{inner}</a>'


def process_chart() -> str:
    """The investment lifecycle as one chart: stages 0-4 before launch, stage 5 after. Every stage links to its page."""
    val, rd, tk = get_value(), get_readiness(), default_tracking()
    c = val["costs"]
    top = match(load_use_cases(), EXAMPLE_CLIENTS["Mid-size factory"]).iloc[0]["use_case"]
    n_fix = len([1 for _, s in rd["weakest"] if s < 3.5][:2]) if rd["verdict"] != "Go" else 0
    arrow = '<span class="arrow">→</span>'
    before = arrow.join([
        flow_node(None, "0", "Define the business problem", "Quantify unplanned downtime and its impact.",
                  f"{money(c.failures_per_year * c.cost_missed)} lost a year", "Baseline"),
        flow_node("pick", "1", "Evaluate AI solutions", "Compare 12 options on value, ease and data readiness.",
                  top, "Selected"),
        flow_node("value", "2", "Build the business case", "Estimate savings, costs, payback and risks.",
                  f"{money(val['annual_net'])} a year", "Projection"),
        flow_node("ready", "3", "Assess readiness", "Find adoption barriers and process gaps.",
                  f"{n_fix} gaps to fix" if n_fix else "Ready to go", "Assessment"),
        flow_node("case", "4", "Prepare the decision", "Recommendation, evidence, risks and decision rule.",
                  decision_rule(val, rd)[0], "Recommendation"),
    ])
    after = flow_node("track", "5", "Measure realized value", "Compare actual results with the forecast.",
                      f"{tk['pct']*100:.0f}% of year-1 target", "Simulated result")
    return ('<div class="flow">'
            f'<div class="lane"><div class="lane-label">Before launch</div><div class="lane-nodes">{before}</div></div>'
            '<div class="launch"><span>Launch</span><em>stages 0–4 happen before launch, stage 5 after</em></div>'
            f'<div class="lane"><div class="lane-label">After launch</div><div class="lane-nodes">{after}'
            '<div class="loop">Every month: compare savings with the forecast, find the reasons for any gap, act on '
            'them, and feed what was learned back into the business case.</div></div></div>'
            '</div>')


def genai_cards() -> str:
    pilot = load_pilot()
    s1, s2, g2 = pilot["summary"][1], pilot["summary"][2], pilot["runs"][2]
    real = int((g2["true_mode"] != "NONE").sum())
    sc = load_agents()["scores"]["v3"]
    return ('<div class="lane-nodes genai-nodes">'
            + flow_node("pilot", "A", "GenAI work orders",
                        "Can GenAI turn a warning into an accurate draft work order?",
                        f"Right cause: {round(s1['correct_on_real_failures'] * real)} → "
                        f"{round(s2['correct_on_real_failures'] * real)} of {real} breakdowns",
                        f"Measured on {len(g2)} cases")
            + flow_node("agents", "B", "GenAI business-case review",
                        "Can GenAI flag errors in a financial business case?",
                        f"{sc['redteam_caught']} of {sc['redteam_flaws']} seeded errors caught",
                        "Measured on my own test cases")
            + '</div>')


@st.cache_data
def data_facts() -> dict:
    df = load_data()
    return {"rows": len(df), "failures": int(df["Machine failure"].sum()),
            "rate": df["Machine failure"].mean(), "types": df["Type"].value_counts().to_dict()}


def case_and_data() -> str:
    """The three questions a reader asks first: what is the business, why this AI, and what data is behind it."""
    c, f = get_value()["costs"], data_facts()
    res = get_model_results()
    cards = [
        ("The business",
         f"A mid-size parts maker running milling machines. About {c.failures_per_year:.0f} times a year a machine "
         f"breaks down without warning: the line stops, parts are scrapped and a technician is called out. At about "
         f"{money(c.cost_missed)} each, that is {money(c.failures_per_year * c.cost_missed)} a year, the biggest "
         "avoidable cost on the shop floor."),
        ("Why predictive maintenance",
         "Breakdowns rarely come from nowhere: tool wear builds up, torque climbs, heat stops escaping. The machines "
         "already record these readings, so a model can learn the warning signs and flag a machine early enough to "
         "repair it at the next planned stop. Stage 1 compares it with 11 other AI options on value, ease and data "
         "readiness; it is the only one that targets unplanned downtime."),
        ("The data",
         f"<b>{f['rows']:,} machine records</b> from the AI4I 2020 predictive maintenance dataset (UCI). Each record "
         "has five sensor readings: air and process temperature, rotation speed, torque and tool wear, plus the "
         f"product quality (low / medium / high). <b>{f['failures']} records ({f['rate']:.1%}) end in a breakdown</b>, "
         f"from five causes: tool wear, heat, power, overstrain and random. The model learns from "
         f"{res['n_train']:,} records and is tested on {res['n_test']:,} it never saw. The dataset is synthetic, "
         "designed to reflect real industrial machine data."),
    ]
    return ('<div class="case-cards">' + "".join(
        f'<div class="case-card"><div class="row-label">{t}</div><div class="case-text">{x}</div></div>'
        for t, x in cards) + '</div>')


def page_home():
    val = get_value()
    c = val["costs"]
    with st.container(key="hero"):
        st.markdown(
            '<div class="eyebrow">AI business systems analysis · interactive case study</div>'
            f'<div class="hero-title">From {esc_html(money(c.failures_per_year * c.cost_missed).replace(".00M", "M"))} in downtime losses '
            'to a measurable AI investment</div>'
            '<div class="hero-text">How a manufacturing problem becomes an AI investment decision: defining the '
            'problem, evaluating solutions, modelling ROI, preparing for launch and measuring results. '
            'Select any stage to see the analysis and deliverables behind it.</div>'
            '<div class="hero-note">Illustrative case: real machine data and real AI tests; plant costs are '
            'assumptions and the results after launch are simulated.</div>', unsafe_allow_html=True)
        st.markdown(process_chart(), unsafe_allow_html=True)

    with st.container(key="case_data"):
        st.markdown(case_and_data(), unsafe_allow_html=True)

    with st.container(key="role"):
        r1, r2 = st.columns([1.15, 1], gap="large")
        r1.markdown(
            '<div class="row-label">My role</div>'
            '<div class="role-text">Independent project, built end to end by me: the business analysis, the '
            'financial model, the data pipeline, the AI tests and this app.</div>', unsafe_allow_html=True)
        r2.markdown(
            '<div class="row-label">Methods</div>'
            + "".join(f'<span class="chip soft">{m}</span>' for m in [
                "Requirements and user stories", "As-is / to-be process maps", "Decision matrix",
                "ROI and sensitivity model", "Readiness and risk assessment", "Acceptance tests and traceability",
                "AI output testing", "Value tracking"])
            + '<div class="row-label tools">Tools</div><div class="tools-text">Python, pandas, scikit-learn, Plotly, '
              'Streamlit, Amazon Bedrock, Amazon S3, Databricks, pytest</div>', unsafe_allow_html=True)

    st.markdown('<div class="row-label">Look closer at a stage</div>', unsafe_allow_html=True)
    stage = st.segmented_control("Stage", STAGES, default=STAGES[2], key="tour",
                                 label_visibility="collapsed") or STAGES[2]
    tc = tour_content(stage)
    with st.container(key="tour_panel"):
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.markdown(f'<div class="tour-q">{esc_html(tc["q"])}</div>', unsafe_allow_html=True)
            st.markdown("".join(
                f'<div class="tour-label">{label}</div><div class="tour-text">{esc_html(text)}</div>'
                for label, text in [("How I analysed it", tc["how"]), ("What I produced", tc["made"]),
                                    ("Decision it supported", tc["decision"])]), unsafe_allow_html=True)
            if tc["docs"]:
                st.markdown(" · ".join(f"[{d}]({DOCS[d]})" for d in tc["docs"]))
            if tc["page"]:
                st.page_link(PAGES[tc["page"]], label="View analysis and deliverables", icon=":material/arrow_forward:")
        with right:
            st.markdown(f'<div class="chart-title">{esc_html(tc["chart_title"])}</div>', unsafe_allow_html=True)
            st.plotly_chart(tc["fig"], width="stretch", config={"displayModeBar": False, "staticPlot": True})
            st.markdown(f'<div class="evidence">{esc_html(tc["evidence"])}</div>', unsafe_allow_html=True)

    with st.container(key="genai"):
        st.markdown(
            '<div class="row-label">Supporting GenAI experiments</div>'
            '<div class="genai-intro">Separate from the main case. The breakdown warnings above come from a '
            '<b>predictive</b> model that scores sensor data. These two experiments test <b>generative</b> AI, which '
            'writes text, with graded results and a person reviewing the output.</div>' + genai_cards(),
            unsafe_allow_html=True)

    with st.container(key="conclusion"):
        st.markdown(
            '<div class="row-label">What this case study demonstrates</div>'
            '<div class="conclusion">An AI project needs more than a good model: a clearly defined business problem, '
            'a defensible investment case, a team ready to use it, measurable acceptance criteria, and a regular '
            'check that the promised value is real.</div>'
            '<div class="doc-links">' + "".join(f'<a href="{u}" target="_blank">{esc_html(d)}</a>'
                                                for d, u in DOCS.items()) + '</div>', unsafe_allow_html=True)
        with st.expander("Data, assumptions and limitations"):
            st.markdown(
                "- **Real:** the 10,000 machine records (AI4I 2020, UCI) and the model trained on them; the Amazon "
                "Bedrock tests; the S3 → Databricks data pipeline.\n"
                "- **Assumed, and editable in the app:** the factory's costs, the scores of the 12 AI options and the "
                "readiness ratings.\n"
                "- **Simulated:** the 12 months after launch.\n"
                "- **Limits:** the dataset is synthetic, so the model's accuracy is likely better than in a real plant. "
                "The GenAI tests are small (30 and 19 cases) and the business-case test was written by me, so it is "
                "not blind. Stakeholders are personas.")
    nav_footer("home")


# ---------------------------------------------------------------- Style and navigation
st.markdown("""
<style>
:root { --teal: #0f6d74; --green: #1d7a45; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
[data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 2.5rem; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * { animation: fadeUp 0.5s ease-out both; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(2) { animation-delay: 0.05s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(3) { animation-delay: 0.1s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(4) { animation-delay: 0.15s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(n+5) { animation-delay: 0.2s; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
[data-testid="stMainBlockContainer"] h1 { font-size: 2.1rem; line-height: 1.2; padding: 0.1rem 0 0.3rem; }
.eyebrow, .row-label, .box-label, .story-step, .kpi-label {
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.74rem; font-weight: 700; color: var(--teal); }
.row-label { margin-top: 0.6rem; opacity: 0.85; }
.intro { font-size: 1.05rem; opacity: 0.8; max-width: 52rem; }
.box-label { margin-bottom: 0.2rem; }
[class*="st-key-answer_"] { background: rgba(15, 109, 116, 0.08); border-left: 4px solid var(--teal);
                            border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 1.1rem; }
[class*="st-key-next_"] { background: rgba(29, 122, 69, 0.06); border-left: 4px solid var(--green);
                          border-radius: 10px; padding: 0.9rem 1.2rem; }
[class*="st-key-next_"] .box-label { color: var(--green); }
.kpi-label { color: inherit; opacity: 0.65; font-weight: 600; }
.kpi-value { font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin: 0.15rem 0; }
.kpi-note { font-size: 0.85rem; opacity: 0.7; }
[class*="st-key-kpi_"] { height: 100%; }
[class*="st-key-kpi_"] { background: #fff; border-color: transparent !important; border-radius: 20px;
                         box-shadow: 0 4px 16px rgba(29, 37, 33, 0.06); }
.st-key-hero { background: linear-gradient(135deg, #e3ebdd 0%, #f1f5ec 55%, #eef6cf 100%); border-radius: 32px;
               padding: 2.2rem 2.4rem 1.8rem; }
.hero-title { font-size: 2.55rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; color: #1d2521;
              margin: 0.4rem 0 0.8rem; }
.hero-text { font-size: 1.08rem; line-height: 1.55; color: #3d4741; margin-bottom: 0.9rem; }
.chip { display: inline-block; background: #fff; border-radius: 999px; padding: 0.35rem 0.85rem; margin: 0 0.35rem 0.45rem 0;
        font-size: 0.88rem; color: #1d2521; box-shadow: 0 1px 3px rgba(29, 37, 33, 0.08); }
.chip.soft { background: #eef2ea; box-shadow: none; }
.flow { margin-top: 1.4rem; }
.lane { display: flex; gap: 1rem; align-items: stretch; margin-bottom: 0.4rem; }
.lane-label { flex: 0 0 6.5rem; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.68rem; font-weight: 700;
              color: #56615a; padding-top: 1rem; }
.lane-nodes { flex: 1; display: flex; align-items: stretch; gap: 0.35rem; flex-wrap: nowrap; }
.node { flex: 1 1 0; min-width: 0; background: #fff; border-radius: 18px; padding: 0.8rem 0.85rem; display: flex;
        flex-direction: column; gap: 0.2rem; text-decoration: none !important; color: #1d2521 !important;
        box-shadow: 0 2px 10px rgba(29, 37, 33, 0.06); transition: transform 0.2s ease, box-shadow 0.2s ease; }
a.node:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(29, 37, 33, 0.12); }
.node.need { background: transparent; box-shadow: none; border: 1.5px dashed #b9c4b4; }
.node .num { width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #1d2521; color: #fff; font-size: 0.75rem;
             font-weight: 700; display: flex; align-items: center; justify-content: center; }
.node.need .num { background: #b9c4b4; }
.node-top { display: flex; justify-content: space-between; align-items: center; }
.node .go { color: #0f6d74; font-weight: 700; opacity: 0.55; transition: opacity 0.2s ease, transform 0.2s ease; }
a.node:hover .go, a.node:focus-visible .go { opacity: 1; transform: translateX(3px); }
a.node:focus-visible { outline: 3px solid #0f6d74; outline-offset: 2px; }
.node-kind { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #56615a;
             margin-top: 0.35rem; }
.node-title { font-weight: 700; font-size: 0.95rem; line-height: 1.2; margin-top: 0.2rem; }
.node-q { font-size: 0.8rem; color: #4d5852; line-height: 1.3; flex: 1; }
.node-res { align-self: flex-start; background: #d4ee5e; border-radius: 10px; padding: 0.15rem 0.55rem; line-height: 1.35; font-size: 0.8rem;
            font-weight: 600; margin-top: 0.15rem; }
.arrow { align-self: center; color: #8a979c; font-size: 1.1rem; }
.launch { margin: 1.1rem 0 1.1rem 7.5rem; border-top: 2px solid #1d2521; position: relative; }
.launch em { position: absolute; top: -0.65rem; left: 5.6rem; background: #eef3e6; padding: 0 0.5rem; font-size: 0.75rem;
             color: #4d5852; font-style: normal; }
.launch span { position: absolute; top: -0.75rem; left: 0; background: #1d2521; color: #fff; border-radius: 999px;
               padding: 0.1rem 0.7rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.loop { flex: 3 1 0; align-self: center; font-size: 0.9rem; color: #3d4741; padding: 0 0.8rem; }
.hero-note { font-size: 0.85rem; color: #4d5852; border-left: 3px solid #b9c4b4; padding-left: 0.7rem; }
.st-key-role { padding: 1.2rem 0.4rem 0.4rem; }
.st-key-case_data { margin-top: 1.2rem; }
.case-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.case-card { background: #fff; border-radius: 22px; padding: 1.1rem 1.3rem; box-shadow: 0 4px 16px rgba(29, 37, 33, 0.06); }
.case-text { font-size: 0.93rem; line-height: 1.55; color: #2f3833; margin-top: 0.35rem; }
@media (max-width: 800px) { .case-cards { grid-template-columns: 1fr; } }
.role-text { font-size: 1rem; line-height: 1.55; color: #2f3833; margin-top: 0.3rem; }
.row-label.tools { margin-top: 0.4rem; }
.tools-text { font-size: 0.9rem; color: #3d4741; }
.tour-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #0f6d74;
              margin-top: 0.7rem; }
.tour-text { font-size: 0.97rem; line-height: 1.5; }
.evidence { font-size: 0.82rem; color: #4d5852; background: #f1f4ed; border-radius: 10px; padding: 0.4rem 0.7rem; }
.st-key-genai { margin-top: 1.2rem; background: #f1f4ed; border: 1.5px dashed #b9c4b4; border-radius: 28px;
                padding: 1.4rem 1.6rem; }
.genai-intro { font-size: 0.95rem; color: #2f3833; max-width: 52rem; margin: 0.2rem 0 0.9rem; }
.genai-nodes { max-width: 60%; gap: 0.8rem; }
.st-key-conclusion { margin-top: 1.2rem; }
.conclusion { font-size: 1.2rem; line-height: 1.5; font-weight: 600; max-width: 54rem; margin: 0.3rem 0 0.8rem;
              border-left: 4px solid #d4ee5e; padding-left: 1rem; }
.doc-links a { display: inline-block; margin: 0 0.4rem 0.5rem 0; padding: 0.35rem 0.85rem; border-radius: 999px;
               background: #fff; color: #1d2521 !important; text-decoration: none; font-size: 0.88rem;
               box-shadow: 0 1px 3px rgba(29, 37, 33, 0.1); }
.doc-links a:hover { background: #eef4d6; }
@media (max-width: 800px) {
  .lane { flex-direction: column; gap: 0.3rem; } .lane-label { flex: none; padding-top: 0; }
  .lane-nodes { flex-direction: column; } .arrow { transform: rotate(90deg); }
  .launch { margin-left: 0; } .launch em { display: none; } .genai-nodes { max-width: none; }
  .loop { padding: 0.4rem 0; } }
.st-key-tour_panel { background: #fff; border-radius: 28px; padding: 1.6rem 1.9rem; box-shadow: 0 12px 36px rgba(29, 37, 33, 0.07); }
.tour-q { font-size: 1.5rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.01em; margin: 0.3rem 0 0.6rem; }
.chart-title { font-size: 0.85rem; color: #6b766d; margin-bottom: 0.2rem; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; padding: 0.35rem 1rem; }
@media (max-width: 640px) { .hero-title { font-size: 1.9rem; } .st-key-hero { padding: 1.4rem 1.2rem; border-radius: 24px; } }
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stPageLink"] a { border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 8px; padding: 0.3rem 0.8rem;
                               transition: background 0.2s ease, border-color 0.2s ease; }
[data-testid="stPageLink"] a:hover { border-color: var(--teal); background: rgba(15, 109, 116, 0.08); }
.stAppDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

PAGES = {
    "home": st.Page(page_home, title="Overview", icon=":material/home:", default=True),
    "pick": st.Page(page_pick, title="1 · Evaluate AI solutions", url_path="match"),
    "value": st.Page(page_value, title="2 · Build the business case", url_path="value"),
    "ready": st.Page(page_ready, title="3 · Assess readiness", url_path="readiness"),
    "case": st.Page(page_case, title="4 · Prepare the decision", url_path="business-case"),
    "track": st.Page(page_track, title="5 · Measure realized value", url_path="track"),
    "pilot": st.Page(page_pilot, title="A · GenAI work orders", url_path="genai-pilot"),
    "agents": st.Page(page_agents, title="B · GenAI business-case review", url_path="agents"),
}
st.navigation({"": [PAGES["home"]],
               "Before launch": [PAGES[k] for k in ("pick", "value", "ready", "case")],
               "After launch": [PAGES["track"]],
               "GenAI experiments": [PAGES["pilot"], PAGES["agents"]]}, position="top").run()
