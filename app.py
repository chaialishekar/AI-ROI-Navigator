"""AI ROI Navigator: an AI business systems analysis case study

An AI provider sells a library of AI tools; its clients ask which one to buy first and whether it will pay off.
Evaluate options compares the library for one client in dollars; the later phases take the top pick for an
auto-parts plant, predictive maintenance, through to a decision and a check of the value it delivers after launch.

Start              - one card per tool (the story is on the case-study website, site/)
  Evaluate options   - every AI tool for the client's industry, priced for them, with ranges and evidence
  Business case      - a real predictive maintenance model, priced in dollars, with ranges and sensitivity
  Readiness          - readiness check, stakeholder impact, human review and escalation rules
  Decide             - recommendation, decision rule and a two-page decision brief
  Measure            - forecast vs actual savings, reasons for the gap (fed by the S3 -> Databricks pipeline)
Side experiments (separate from the main case)
  A GenAI work orders          - measured Bedrock pilot of an AI work-order writer, and its ROI
  B GenAI business-case review - intake, memo and critic agents on Bedrock, graded on known answers

Every page follows one template: the question, the result, three numbers, one chart and what to do next,
then the inputs and method, then Back / Next links.
"""

import html
import json
import os
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.brief import build_brief
from src.model import train_and_score
from src.library import (EVIDENCE_ORDER, EXAMPLE_CLIENTS, Client, break_even_improvement, evidence_strength,
                         load_library, rank, rank_changes, too_close)
from src.readiness import ACTIONS, DIMENSIONS, verdict
from src.genai import COST_LABELS, GenAIInputs
from src.genai import evaluate as genai_evaluate
from src.tracking import ADOPTION_FLOOR, COLUMNS, DRIFT_DROP, DRIFT_WINDOW, explain_gap, month_label, track, validate
from src.agents import ASSUMPTIONS as AGENT_ASSUMPTIONS, INTAKE_FIELDS, run_live
from src.agents import MODEL_ID as AGENT_MODEL_ID
from agents.grade import scorecard
from src.workorder import MODEL_ID, MODES, grade, parse, rule_alerts, summary as pilot_summary
from src.value import (
    ACCURACY_THRESHOLD,
    ASSUMPTIONS,
    Costs,
    alert_threshold,
    bootstrap,
    default_ranges,
    evaluate,
    plain_language,
    scenarios,
    sensitivity,
    sweep,
)

st.set_page_config(page_title="AI ROI Navigator", page_icon=":material/insights:", layout="wide")


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
def load_library_data() -> pd.DataFrame:
    return load_library()



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
STRENGTH_COLORS = {"Stronger": "#1d7a45", "Medium": "#0f6d74", "Weaker": "#a25f00"}


def pick_client(detail) -> Client:
    """Client chooser at the top of Evaluate options; their figures can be edited under 'Show the details'."""
    name = st.segmented_control("Client", list(EXAMPLE_CLIENTS), default=list(EXAMPLE_CLIENTS)[0],
                                key="client_pick") or list(EXAMPLE_CLIENTS)[0]
    base = EXAMPLE_CLIENTS[name]
    with detail.container():
        st.markdown(f"**{base.name}'s figures.** Example figures: replace them with the client's own.")
        a, b = st.columns(2)
        quality = a.slider("Data quality", 1, 5, base.data_quality, key=f"dq_{name}",
                           help="1 = scattered spreadsheets, 5 = clean, connected data. Each point short of what a "
                                "solution needs adds 20% to its setup cost and 2 months to go live.")
        size = b.number_input("Deployment size (sites or lines)", 1, 20, base.size, key=f"size_{name}",
                              help="The provider's price list is per site; setup and running costs are multiplied by this.")
        costs = st.data_editor(pd.DataFrame({"Problem": list(base.problem_costs),
                                             "Cost a year today ($)": list(base.problem_costs.values())}),
                               hide_index=True, width="stretch", disabled=["Problem"], key=f"costs_{name}",
                               column_config={"Cost a year today ($)": st.column_config.NumberColumn(
                                   min_value=0, step=50_000, format="dollar")})
    return replace(base, data_quality=quality, size=size,
                   problem_costs=dict(zip(costs["Problem"], costs["Cost a year today ($)"])))


def range_chart(r: pd.DataFrame, basis: str) -> go.Figure:
    """3-year net value per solution: a bar from the low to the high end of the published range, a dot on the
    value the ranking uses, coloured by how strong the evidence is."""
    colors = [STRENGTH_COLORS[evidence_strength(e)] for e in r["evidence"]]
    fig = go.Figure(go.Bar(y=r["solution"], x=r["net3_high"] - r["net3_low"], base=r["net3_low"], orientation="h",
                           marker=dict(color=colors, opacity=0.22, cornerradius=8), hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=r["solution"], x=r["net_3y"], mode="markers+text", marker=dict(size=14, color=colors),
                             text=[money(v) for v in r["net_3y"]], textposition="top center", cliponaxis=False,
                             hovertemplate="%{y}: %{text} over 3 years (" + basis + ")<extra></extra>"))
    fig.add_vline(x=0, line_color="#8a979c", line_width=1)
    fig.update_layout(height=90 + 70 * len(r), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(tickformat="$~s", title="3-year net value"),
                      yaxis=dict(autorange="reversed", title=""))
    return fig


def page_pick():
    answer, client, detail = page_header(1, "Which AI should the client buy first?",
                                         "Every AI tool in the provider's library that fits the client's industry, "
                                         "priced for this client in dollars, with the published range behind each "
                                         "number and how strong that evidence is.")
    library = load_library_data()
    with client:
        cl = pick_client(detail)
        st.markdown(f'<div class="intro">{esc_html(cl.blurb)} · data quality {cl.data_quality}/5 · '
                    f'{cl.size} site{"s" if cl.size > 1 else ""}.</div>', unsafe_allow_html=True)
        view = st.segmented_control("Rank on", ["Expected", "Cautious"], default="Expected", key="lib_view",
                                    help="Expected: the middle of each published range. Cautious: the low end.") or "Expected"
    cautious = view == "Cautious"
    r = rank(library, cl, cautious=cautious)
    top, second = r.iloc[0], r.iloc[1] if len(r) > 1 else None
    st.session_state["library_client"], st.session_state["library_pick"] = cl.name, top["solution"]
    basis = "low end of the published range" if cautious else "middle of the published range"
    close = too_close(r)
    lead = (f"**Start with {top['solution'].lower()}.** "
            f"{money(top['net_3y'])} over 3 years ({basis}), pays back in {months(top['payback'])}, "
            f"evidence: {top['evidence'].lower()}.")
    if close and second is not None:
        lead += (f" {second['solution']} is close on value ({money(second['net_3y'])}), so payback and evidence decide: "
                 + ("the top pick is ahead on both." if top["payback"] <= second["payback"]
                    and top["evidence_rank"] <= second["evidence_rank"] else "compare them in the table below."))
    answer.markdown(md(lead))

    changes = rank_changes(library, cl)
    weak = [x for x in r.head(2).itertuples() if evidence_strength(x.evidence) == "Weaker"]
    with client:
        kpis([("AI tools for this client", f"{len(r)} of {len(library)}", f"the library's {cl.industry.lower()} entries"),
              ("Start with", top["solution"], f"{money(top['net_3y'])} over 3 years"),
              ("It only needs to fix", f"{break_even_improvement(top)*100:.0f}%",
               f"of {top['problem'].lower()} to pay back")])
        st.markdown("**3-year net value for each AI tool:** the bar is the published range, the dot the value "
                    "ranked on &nbsp; " + " &nbsp; ".join(
            f'<span style="color:{c}">●</span> {s.lower()} evidence' for s, c in STRENGTH_COLORS.items()),
            unsafe_allow_html=True)
        st.plotly_chart(range_chart(r, view.lower()), width="stretch")
        steps = [f"**Buy in this order as budget allows.** The tools fix different problems, so this is an order "
                 f"of priority, not a single winner."]
        steps.append(("**This order differs from the expected view:** " if cautious else
                      "**The order changes on the cautious view:** ") + "; ".join(changes)
                     + (". Expected → cautious rank shown." if cautious else ". Check that view before committing.")
                     if changes else "**The order holds on the cautious view**, when every tool only "
                     "reaches the low end of its published range.")
        steps += [f"**Treat {x.solution.lower()}'s number with care:** it rests on {x.evidence.lower()} figures."
                  for x in weak]
        if top["solution"] == "Predictive maintenance":
            steps.append("**Next:** the following phases take predictive maintenance all the way through for the "
                         "auto-parts plant: a test on public machine data, a readiness check, sign-off and tracking.")
        elif evidence_strength(top["evidence"]) == "Weaker" and not cautious and changes:
            cautious_top = rank(library, cl, cautious=True).iloc[0]["solution"]
            steps.append(f"**Next:** test the claim behind {top['solution'].lower()} in a short pilot before a full "
                         f"business case. On cautious numbers {cautious_top.lower()} leads, so it is the safer first buy.")
        else:
            steps.append(f"**Next:** a full business case for {top['solution'].lower()}, following the same steps as "
                         "the predictive maintenance example.")
        next_steps([md(x) for x in steps], label="What this means")
        if top["solution"] == "Predictive maintenance":
            st.page_link(PAGES["value"], label="Open the predictive maintenance business case",
                         icon=":material/arrow_forward:")

    t_all, t_src, t_how = detail.tabs(["Every number", "Evidence and sources", "How it's worked out"])
    with t_all:
        st.dataframe(pd.DataFrame({
            "Rank": r["rank"], "AI tool": r["solution"], "Problem it fixes": r["problem"],
            "Problem costs a year": r["problem_cost"].map(money),
            "Published improvement": [f"{lo:.0%}–{hi:.0%}" for lo, hi in zip(r["improvement_low"], r["improvement_high"])],
            "Value a year": [money_range((lo, hi)) for lo, hi in zip(r["value_low"], r["value_high"])],
            "Setup": r["setup"].map(money), "Running a year": r["running"].map(money),
            "Months to go live": r["months"].map(lambda m: f"{m:.0f}"),
            "Payback": r["payback"].map(months), "3-year ROI": r["roi"].map(lambda x: f"{x*100:,.0f}%"),
            "Break-even": [f"{break_even_improvement(x)*100:.0f}%" for _, x in r.iterrows()],
            "Evidence": r["evidence"]}), hide_index=True, width="stretch")
        st.caption("Break-even: the share of the problem the tool must remove just to pay back its 3-year cost. "
                   "Setup and months include extra data work where the client's data falls short.")
    with t_src:
        st.dataframe(r[["solution", "what_it_does", "evidence", "source", "source_url"]].rename(columns={
            "solution": "AI tool", "what_it_does": "What it does", "evidence": "Evidence", "source": "What was found",
            "source_url": "Source"}), hide_index=True, width="stretch",
            column_config={"Source": st.column_config.LinkColumn(display_text="Open")})
        st.caption("Evidence, strongest first: " + " > ".join(EVIDENCE_ORDER) + ". Where a source gave a single "
                   "figure, the range around it is my estimate and is kept below vendor claims.")
    with t_how:
        st.markdown(md(
            "Three kinds of number go in:\n"
            "- **The client's figures:** what each problem costs them a year (example figures, editable above).\n"
            "- **Published results:** how much of that cost each tool removed elsewhere, low to high, with the source.\n"
            "- **The provider's price list:** setup, running cost and months to go live per site (illustrative).\n\n"
            "For each tool: value a year = problem cost × improvement; net a year = value − running cost; "
            "3-year net value = net a year × months live in 3 years ÷ 12 − setup; payback = months to go live + "
            "setup ÷ monthly net.\n\n"
            "**What keeps the comparison fair**\n"
            "1. Tools are only compared for one client, never across industries.\n"
            "2. The same formula, 3-year window and cost treatment for every tool.\n"
            "3. Every number is a range; a tool that wins only on optimistic numbers shows up on the cautious view.\n"
            "4. Every number is labelled with the strength of its evidence.\n"
            "5. Missing data costs extra: each point the client's data falls short adds 20% to setup and 2 months.\n"
            "6. Break-even is shown, since a low bar to clear is safer than a high ROI.\n\n"
            "Code: `src/library.py`, data: `data/ai_library.csv`."))
    nav_footer("pick")


# ---------------------------------------------------------------- Page 2
def page_value():
    answer, client, detail = page_header(2, "Is it worth it?",
                                         "A closer look at the top pick for the auto-parts plant: predictive "
                                         "maintenance, which warns before a machine breaks down, tested on public (synthetic) machine "
                                         "data. Every breakdown it catches, misses or wrongly predicts is turned into dollars.")
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
        st.caption("Defaults are illustrative assumptions for the auto-parts plant. Edit them to match your operation.")

    with t_acc:
        model_name = st.radio("Model", list(res["models"].keys()), index=1, horizontal=True, key="model_pick")
        proba = res["models"][model_name]["proba"]
        curve = sweep(y_test, proba, costs)
        # Both reference levels come from the costs or convention, not the test set, so the test stays an honest check
        best_money = evaluate(y_test, proba, alert_threshold(costs), costs)
        best_acc = evaluate(y_test, proba, ACCURACY_THRESHOLD, costs)
        threshold = st.slider("Alert threshold: alert when failure probability is at least…",
                              0.01, 0.99, float(best_money["threshold"]), 0.01, key="thr",
                              help="Lower = more sensitive: catches more failures but raises more false alarms.")
    r = evaluate(y_test, proba, threshold, costs)
    band = bootstrap(y_test, proba, threshold, costs)
    gap = best_money["annual_net"] - best_acc["annual_net"]

    with t_acc:
        if gap > 1:
            st.markdown(md(
                f"**Accuracy is not value.** The usual 50% cut-off ({best_acc['accuracy']*100:.1f}% accurate) saves "
                f"{money(best_acc['annual_net'])} a year. The break-even level ({best_money['threshold']:.2f}) saves "
                f"{money(best_money['annual_net'])}: alert whenever the chance of failure makes an inspection worth it: "
                f"{money(costs.cost_false_alarm)} ÷ ({money(costs.cost_missed - costs.cost_caught)} saved per early "
                f"catch + {money(costs.cost_false_alarm)}). It is set from the costs, not tuned on the test data."))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["annual_net"], name="Annual net savings ($)",
                                  line=dict(color="#0f6d74", width=3)))
        fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["accuracy"] * 100, name="Accuracy (%)",
                                  line=dict(color="#a25f00", dash="dot"), yaxis="y2"))
        fig2.add_vline(x=best_money["threshold"], line_color="#1d7a45", line_dash="dash",
                       annotation_text="break-even", annotation_position="top left")
        fig2.add_vline(x=best_acc["threshold"], line_color="#a25f00", line_dash="dash",
                       annotation_text="50% cut-off", annotation_position="top right")
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
            f"{int(y_test.sum())} actual failures, so the rates are estimates."))

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
            b = evaluate(y_test, m["proba"], alert_threshold(costs), costs)
            rows.append({"Model": name, "AUC": round(m["auc"], 3), "Alert level": round(b["threshold"], 2),
                         "Recall": f"{b['recall']*100:.0f}%", "Precision": f"{b['precision']*100:.0f}%",
                         "Annual net savings": money(b["annual_net"]), "Payback (months)": round(b["payback_months"], 1)})
        st.markdown("**Models compared**, both at the break-even alert level")
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
                   f"({int(y_test.sum())} actual failures). Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0). "
                   "Dollar results scale the test-set rates to your plant's failures per year.")

    st.session_state["value"] = {**r, "model": model_name, "band": band, "cases": cases, "costs": costs,
                                 "top_driver": top["assumption"].replace(" ($)", ""), "top_swing": top["swing"]}
    robust = low["annual_net"] > 0 and low["payback_months"] <= 36
    pub = published_estimate(costs)
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
            md(f"**Plan with the published figure.** Studies report {pub['low_pct']:.0%}–{pub['high_pct']:.0%} less "
               f"downtime, which gives {money(pub['mid'])} a year here (see Evaluate options). This test gives "
               f"{money(r['annual_net'])} because the synthetic data makes breakdowns easier to predict than a real "
               "plant would. Approve on the published figure and prove the rest in the pilot."),
            md(f"**Go live soon.** Every month of delay costs about {money(r['annual_net'] / 12)}."),
        ])
    nav_footer("value")


def published_estimate(costs: Costs) -> dict:
    """Net savings a year for predictive maintenance from the published downtime reductions (the library in Evaluate options),
    applied to this plant's breakdown cost and running cost."""
    row = load_library_data().query("solution == 'Predictive maintenance' and industry == 'Automotive'").iloc[0]
    lo, hi = row["improvement_low"], row["improvement_high"]
    downtime = costs.failures_per_year * costs.cost_missed
    return {"low_pct": lo, "high_pct": hi, "mid": downtime * (lo + hi) / 2 - costs.running_cost}


# ---------------------------------------------------------------- Page 3
EXAMPLE_PILOT = {"Data": [4, 4], "Fits how people work": [2, 2], "Clear payoff": [4, 3],
                 "People trust it": [3, 2], "Clear ownership": [3, 3]}


STAKEHOLDER_IMPACT = [
    ("Maintenance planner", "Checks AI warnings and draft work orders instead of writing orders after a breakdown",
     "Will it bury us in false alarms?", "Warning level set to keep false alarms rare (Business case); planner approves every order"),
    ("Technicians", "Planned repairs at the next stop instead of emergency call-outs", "Will the warnings be right?",
     "Every warning gives its reason in plain words; outcomes are logged each month"),
    ("Operations director", "Owns the monthly value review", "Is it paying off?",
     "The Measure phase compares savings with the forecast every month"),
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


# Which option each decision-rule outcome stands for, in the options table on the Decide page
OPTION_FOR_LABEL = {"Proceed": "Buy the AI and roll it out", "Proceed after fixes": "Pilot on one line after fixes",
                    "Revise": "Wait until the team is ready", "Stop": "Don't buy it"}


def options_considered(val: dict, rd: dict, label: str) -> str:
    """Every option the decision weighed, AI or not, with one line on why it was or wasn't picked."""
    rules = alternatives(val, val["costs"])["rules"]
    worst = val["cases"]["Low"]["annual_net"]
    c = val["costs"]
    weakest = rd["weakest"][0][0].lower() if rd.get("weakest") else "n/a"
    why = {
        "Buy the AI and roll it out": (f"Saves {money(val['annual_net'])} a year as planned, but the team check says "
                                       f"\"{rd['verdict'].lower()}\" (weakest area: {weakest})."
                                       if rd["verdict"] != "Go" else
                                       f"Saves {money(val['annual_net'])} a year and the team is ready."),
        "Pilot on one line after fixes": ("Close the readiness gaps, then prove the savings and the false-alarm rate "
                                          "on one line before scaling."),
        "Use the manual's rules instead": (f"No AI needed and saves {money(rules['annual_net'])} a year, but raises "
                                           f"~{rules['false_alarms']:.0f} false alarms a year against the AI's "
                                           f"~{val['false_alarms']:.0f}, the kind of noise a crew learns to ignore. The rules stay, to explain "
                                           "each AI warning."),
        "Wait until the team is ready": (f"Still saves {money(worst)} a year in the worst case, so waiting forgoes "
                                         f"about {money(val['annual_net'] / 12)} a month."
                                         if worst > 0 else "Savings turn negative if costs are worse than expected."),
        "Don't buy it": (f"Breakdowns keep costing the plant {money(c.failures_per_year * c.cost_missed)} a year, "
                         "and the evidence for the AI is medium, not weak."
                         if val["annual_net"] > 0 else "Expected savings are not positive."),
    }
    chosen = OPTION_FOR_LABEL.get(label)
    return ("| Option | Verdict | Why |\n|---|---|---|\n" + "".join(
        f"| {'**' + o + '**' if o == chosen else o} | {'**Chosen**' if o == chosen else 'Not chosen'} | {md(w)} |\n"
        for o, w in why.items()))


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
    lib_rank, pub = rank(load_library_data(), EXAMPLE_CLIENTS["Auto-parts plant"]), published_estimate(c)
    return {
        "title": "Predictive maintenance for an auto-parts plant", "client": "Auto-parts plant, tier-1 automotive supplier",
        "author": "Chaitali Shekar", "decision": label, "decision_reason": reason,
        "kpis": [("Projected savings a year", money(val["annual_net"]), "after AI running costs"),
                 ("Pays for itself in", months(val["payback_months"]), f"{money(c.deployment_cost)} to set up"),
                 ("3-year ROI", f"{val['roi_3y']*100:,.0f}%", f"{money(val['three_year'])} net over 3 years"),
                 ("If costs are worse", money(cases["Low"]["annual_net"]), "savings a year, worst case")],
        "situation": (f"An auto-parts plant has about {c.failures_per_year:.0f} unplanned machine breakdowns a year at "
                      f"about {money(c.cost_missed)} each in downtime and emergency repair: <b>{money(val['cost_without'])} "
                      "a year</b>. The machines already record temperature, speed, torque and tool wear, so a model can "
                      f"warn before a breakdown and turn it into a planned repair at about {money(c.cost_caught)}. "
                      f"Of the provider's {len(lib_rank)} AI tools for automotive clients, predictive maintenance ranks "
                      f"first for this plant: payback in {months(lib_rank.iloc[0]['payback'])}, ahead of visual inspection, which rests on a vendor claim. "
                      f"Published results put it at <b>{money(pub['mid'])} a year</b>. The model test below gives more "
                      "on synthetic data, so the budget should be approved on the published figure and the rest proved "
                      "in the pilot."),
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
                     ["Published downtime reduction", "Analyst research", "30-50% less downtime (McKinsey, 2017); 20% used as the cautious low end"],
                     ["Provider prices", "Assumption", "Illustrative price list, the same for every AI tool compared"],
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
        pct_txt = f"{tk['pct']*100:.0f}% of the model's estimate" if np.isfinite(tk["pct"]) else "estimate not positive"
        tracking_md = (
            f"\n## Value delivered after go-live ({tk['source']}, {tk['first']} to {tk['last']})\n"
            f"- Net savings to date: {money(tk['delivered'])} vs the {money(tk['plan'])} approved plan, and "
            f"{money(tk['promised'])} model estimate ({pct_txt})\n"
            + (f"- Biggest shortfall: {tk['worst_cause']} ({money(tk['worst_effect'])}). "
               f"{GAP_ACTIONS[tk['worst_cause']]}\n" if tk["worst_cause"] else "")
            + (f"- Crew acted on fewer than {ADOPTION_FLOOR:.0%} of warnings in: {', '.join(tk['low_adoption_months'])}\n"
               if tk["low_adoption_months"] else f"- Crew acted on at least {ADOPTION_FLOOR:.0%} of warnings every month\n")
            + ("- The AI is getting worse: it caught "
               f"{tk['recall_now']*100:.0f}% of breakdowns over the last {DRIFT_WINDOW} months, {DRIFT_DROP*100:.0f}+ points below launch. Update it with recent data.\n"
               if tk["drift_now"] else "- The AI is catching breakdowns as well as at launch\n")

        )
    summary = f"""# AI Business Case: Predictive Maintenance

**Client:** Auto-parts plant (tier-1 automotive supplier). Main problem: unplanned downtime.
**Recommended solution:** Predictive maintenance, first of the provider's AI tools for this plant (Evaluate options). Last client compared: {st.session_state.get('library_client', 'n/a')} → {st.session_state.get('library_pick', 'n/a')}.

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
Acceptance criteria: 13 user acceptance tests and a requirements traceability matrix (docs/requirements.md).
""" + tracking_md + """
## Key assumptions
- Plant costs as entered in Business case (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
    with client:
        kpis([("Projected savings a year", money(val["annual_net"]), f"pays for itself in {months(val['payback_months'])}"),
              ("If costs are worse", money(val["cases"]["Low"]["annual_net"]), "savings a year, worst case"),
              ("Team ready?", rd["verdict"], f"average {rd['avg']:.1f} / 5")])
        st.markdown("**How the recommendation is decided**")
        st.markdown(rules_md.replace(f"| {label} |", f"| **{label}** (this case) |"))
        st.markdown("**Options considered, AI or not**")
        st.markdown(options_considered(val, rd, label))
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
                         "This is not the model's fault; update failures per year in Business case.",
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


def monthly_vs_forecast(t: pd.DataFrame, ok_color: str, alert_color: str, height: int = 320,
                        plan: float | None = None) -> go.Figure:
    """Actual net savings per month as bars, the model's monthly estimate as a dashed line and, if given, the
    approved plan (a yearly figure) as a dotted line; alert months highlighted."""
    alert = t["adoption_alert"] | t["drift_alert"]
    why = np.where(t["drift_alert"] & t["adoption_alert"], "warnings ignored and AI getting worse",
                   np.where(t["drift_alert"], "AI getting worse", np.where(t["adoption_alert"], "warnings ignored", "on track")))
    fig = go.Figure([
        go.Bar(x=t["label"], y=t["actual"], name="Actual", showlegend=False, marker=dict(color=np.where(alert, alert_color, ok_color),
                                                                         cornerradius=6),
               customdata=why, hovertemplate="%{x}: %{y:$,.0f} (%{customdata})<extra></extra>"),
        go.Scatter(x=t["label"], y=t["forecast"], name="Model estimate", mode="lines", line=dict(color="#15222a", dash="dash", width=2),
                   hovertemplate="Model estimate %{y:$,.0f}<extra></extra>")])
    if plan is not None:
        fig.add_trace(go.Scatter(x=t["label"], y=[plan / 12] * len(t), name="Approved plan", mode="lines",
                                 line=dict(color="#8a979c", dash="dot", width=2),
                                 hovertemplate="Approved plan %{y:$,.0f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(tickformat="$,.0f", title=""),
                      xaxis_title="", legend=dict(orientation="h", y=-0.2), bargap=0.25)
    return fig


def page_track():
    answer, client, detail = page_header(5, "Is it paying off?",
                                         "The one-line pilot passed its checks, so the AI went live across the plant. "
                                         "Every month since, the plant's repair records are turned into dollars and "
                                         "compared with the plan finance approved and with the model's estimate.")
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
                st.markdown(md(f"- Set aside the {month_label(q['month'], pipe['gold']['month'].iloc[0])} file: {q['reason']}"))
        st.caption("The data is a simulated first year after the plant-wide rollout, not real plant data: adoption is "
                   "slow at first, then the model starts to drift. Months are counted from go-live. Columns: month, "
                   "alerts_raised, failures_flagged, failures_prevented, failures_missed, false_alarms.")

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
    plan = published_estimate(c)["mid"] * n / 12     # what the budget was approved on, for the same months
    pct = delivered / promised if promised > 0 else np.nan
    recall_now = t["recall_3m"].dropna().iloc[-1] if t["recall_3m"].notna().any() else np.nan
    tk = st.session_state["tracking"] = summarize_tracking(t, gap, source, c)
    low_months = t.loc[t["adoption_alert"], "label"].tolist()
    drift_months = t.loc[t["drift_alert"], "label"].tolist()

    answer.markdown(md(
        (f"**Yes: {money(delivered)} in {n} months, above the {money(plan)} plan finance approved.** "
         if delivered >= plan else
         f"**Not yet: {money(delivered)} in {n} months, below the {money(plan)} plan finance approved.** ")
        + (f"It reached {pct*100:.0f}% of the model's estimate ({money(promised)}). " if np.isfinite(pct) else "")
        + (f"The biggest reason for the gap: {SHORTFALL[tk['worst_cause']]}. " if tk["worst_cause"] else "")
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
        figa = go.Figure(go.Bar(x=t["label"], y=t["adoption"] * 100,
                                marker_color=["#b4322c" if a else "#1d7a45" for a in t["adoption_alert"]]))
        figa.add_hline(y=ADOPTION_FLOOR * 100, line_dash="dot", line_color="#8a979c",
                       annotation_text=f"{ADOPTION_FLOOR:.0%} target", annotation_position="top left")
        figa.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Alerts acted on (%)",
                           yaxis=dict(range=[0, 105], title=""), xaxis_title="")
        g1.plotly_chart(figa, width="stretch")
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=t["label"], y=t["recall_3m"] * 100, mode="lines+markers", name="Recall (3-month)",
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
            x=["Model estimate"] + gap["cause"].tolist() + ["Actual"],
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
            "exactly to the gap from the model's estimate. The estimate assumes every alert is acted on. Dollar costs per failure can't be "
            f"checked from counts; the biggest cost driver is **{val['top_driver']}**, so check it against real repair invoices."))

    with t_data:
        st.dataframe(t[["label"] + COLUMNS[1:] + ["recall", "adoption", "actual", "forecast"]].round(2)
                     .rename(columns={"label": "month", "forecast": "model_estimate"}), hide_index=True, width="stretch")

    figc = monthly_vs_forecast(t, "#0f6d74", "#c98a2b", height=320, plan=plan * 12 / n)

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
        steps.append(f"**Ask the plant to resend the {month_label(q['month'], pipe['gold']['month'].iloc[0])} file**: "
                     "it failed the data checks.")

    if not np.isfinite(pct) or delivered <= 0:
        call = ("Reconsider", "it has not saved money so far; check whether the problem and the forecast still hold.")
    elif pct >= 0.9:
        call = ("Scale", "it delivers close to the model's estimate; extend it to the company's other plants.")
    else:
        call = ("Improve, then scale", f"it beats the approved plan but delivered {pct*100:.0f}% of the model's "
                                        "estimate; fix the causes below and update the estimate before taking it "
                                        "to other plants." if delivered >= plan else
                                        f"it pays, but is below the approved plan; fix the causes below before taking "
                                        "it to other plants.")
    gap_lines = [f"**{SHORTFALL[cs][0].upper() + SHORTFALL[cs][1:]}**: {money(e)}"
                 for cs, e in zip(gap["cause"], gap["effect"]) if e < 0]
    steps = [md(f"**{call[0]}**: {call[1]}")] + steps

    with client:
        kpis([(f"Saved in {n} months", money(delivered), f"{money(delivered - plan)} vs the approved plan"),
              ("Of the model's estimate", f"{pct*100:.0f}%" if np.isfinite(pct) else "n/a", f"of {money(promised)}"),
              ("Breakdowns caught now", f"{recall_now*100:.0f}%" if np.isfinite(recall_now) else "n/a",
               f"{val['recall']*100:.0f}% at launch")])
        st.markdown("**Saved each month vs the approved plan and the model's estimate** (amber = a month with a warning)")
        st.plotly_chart(figc, width="stretch")
        if gap_lines:
            st.markdown(md(f"**Why it fell short of the model's estimate by {money(promised - delivered)}**  \n"
                           + "  \n".join(gap_lines)))
        st.caption(f"Approved plan: the {money(plan)} finance budgeted for these {n} months, from published studies. "
                   f"Model estimate: the {money(promised)} the model test suggested. Delivered: the same formula applied "
                   "to the plant's monthly records. Simulated data for this example.".replace("$", "\\$"))
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
                                         "drafts the repair order. We tested it on 30 warnings from the synthetic test data, two ways.")
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
        }, index=["Right cause (all 30)", "Right cause (24 true failures)", "False alarm recognised (6)",
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
        f"{s2['correct_on_real_failures']:.0%}, on these 30 synthetic test examples rather than a live plant. "
        f"It mostly frees up time rather than saving cash: {r2['roi']*100:,.0f}% return in year one counting hours "
        f"saved, but {r2['cash_roi']*100:,.0f}% if only {redeploy:.0%} of those hours turn into money. The cost that "
        "matters is the time people spend checking drafts."))

    metrics = ["All 30 warnings", "24 true breakdowns", "6 false warnings"]
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
    """Prove-the-value result at the default plant costs and the break-even alert level."""
    res = get_model_results()
    y, name, costs = res["y_test"], "Gradient boosting", Costs()
    proba = res["models"][name]["proba"]
    thr = alert_threshold(costs)
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


def summarize_tracking(t: pd.DataFrame, gap: pd.DataFrame, source: str, c: Costs) -> dict:
    last = t.iloc[-1]
    delivered, promised = last["cum_actual"], last["cum_forecast"]
    plan = published_estimate(c)["mid"] * len(t) / 12
    worst = gap.loc[gap["effect"].idxmin()]
    return {
        "source": source, "months": len(t), "first": t["label"].iloc[0], "last": last["label"],
        "delivered": delivered, "promised": promised, "pct": delivered / promised if promised > 0 else np.nan,
        "plan": plan, "vs_plan": delivered / plan if plan > 0 else np.nan,
        "adoption_last": last["adoption"], "low_adoption_months": t.loc[t["adoption_alert"], "label"].tolist(),
        "recall_now": t["recall_3m"].dropna().iloc[-1] if t["recall_3m"].notna().any() else np.nan,
        "drift_now": bool(last["drift_alert"]),
        "worst_cause": worst["cause"] if worst["effect"] < 0 else None, "worst_effect": worst["effect"],
    }


def default_tracking() -> dict:
    val, pipe = get_value(), load_pipeline()
    actuals, source = ((pipe["gold"][COLUMNS], "S3 → Databricks pipeline") if pipe
                       else (pd.read_csv("data/sample_actuals.csv"), "simulated sample"))
    t = track(actuals, val, val["costs"])
    return summarize_tracking(t, explain_gap(t, val, val["costs"]), source, val["costs"])



# ---------------------------------------------------------------- Layout helpers
ORDER = ["home", "pick", "value", "ready", "case", "track", "pilot", "agents"]


def esc_html(text: str) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def page_header(step: int, title: str, intro: str, detail_label: str = "Show the details"):
    """Every page: the question, the result, the client view, then the inputs and method.

    Returns (answer, client, detail) containers; the page fills them in any order, so inputs inside the
    detail tabs can be read before the answer and the client view are written."""
    st.title(title)
    answer = st.container(key=f"answer_{step}")
    answer.markdown('<div class="box-label">Result</div>', unsafe_allow_html=True)
    client = st.container()
    detail = st.container(border=True, key=f"detail_{step}")
    detail.markdown(f'<div class="box-label">{esc_html(detail_label)}</div>', unsafe_allow_html=True)
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


# ---------------------------------------------------------------- Start
def break_even_cost(val: dict, c: Costs) -> float:
    """Cost of a breakdown at which the AI stops paying: savings = caught × (breakdown − planned repair) − other costs."""
    if val["caught"] <= 0:
        return float("inf")
    return c.cost_caught + (val["false_alarms"] * c.cost_false_alarm + c.running_cost) / val["caught"]


CASE_STUDY_SITE = "https://chaialishekar.github.io/AI-ROI-Navigator/"
def page_home():
    """Start page: one card per tool."""
    st.title("AI ROI Navigator")
    st.markdown(f'<div class="model-text">Example client: an auto-parts plant. '
                f'<a href="{CASE_STUDY_SITE}" target="_blank">Case study ↗</a></div>', unsafe_allow_html=True)
    tools = [("pick", "Rank the AI tools for a client."),
             ("value", "Savings, payback and ranges from the plant's costs."),
             ("ready", "Rate the team; get a verdict and the gaps."),
             ("case", "Decision rule and the two-page brief (PDF)."),
             ("track", "Monthly results vs the plan and the estimate."),
             ("pilot", "GenAI repair-order drafts, graded on 30 cases."),
             ("agents", "GenAI business-case review, graded on 19 cases.")]
    for row in (tools[:4], tools[4:]):
        for col, (key, what) in zip(st.columns(4), row):
            with col.container(border=True, key=f"tool_{key}"):
                st.page_link(PAGES[key], label=PAGES[key].title, icon=":material/arrow_forward:")
                st.caption(what)
    with st.expander("Data and limitations"):
        st.markdown(
            "- **Data:** 10,000 synthetic machine records (AI4I 2020, UCI).\n"
            "- **Published, with sources:** the results each AI tool got elsewhere.\n"
            "- **Assumed, editable:** the clients' costs, prices and readiness ratings.\n"
            "- **Simulated:** the 12 months after launch.\n"
            "- **Limits:** synthetic data flatters accuracy; the GenAI tests are small (30 and 19 cases).")
    nav_footer("home")


# ---------------------------------------------------------------- Style and navigation
st.markdown("""
<style>
:root { --ink: #111; --muted: #6b6b6b; --faint: #8a8a85; --line: #e6e4df; --teal: #0f6d74; --green: #2f6b1f; }
[data-testid="stMainBlockContainer"] { max-width: 1080px; padding-top: 2.5rem; }
[data-testid="stMainBlockContainer"] h1 { font-size: 2.4rem; font-weight: 400; letter-spacing: -0.03em; line-height: 1.1;
                                          padding: 0.1rem 0 0.4rem; }
[data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 { font-weight: 500;
                                                                                   letter-spacing: -0.015em; }
.box-label, .kpi-label { text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem; font-weight: 600;
                         color: var(--faint); }
.box-label { margin-bottom: 0.2rem; }
.intro, .model-text { color: var(--muted); font-size: 0.98rem; }
.model-text { margin: -0.3rem 0 1rem; }
.model-text a { color: var(--teal) !important; }
[class*="st-key-answer_"] { background: #fff; border: 1px solid var(--line); border-left: 3px solid var(--teal);
                            border-radius: 14px; padding: 0.9rem 1.2rem; font-size: 1.08rem; }
[class*="st-key-next_"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 0.9rem 1.2rem; }
[class*="st-key-detail_"], [class*="st-key-tool_"] { background: #fff; border-radius: 14px; }
.kpi-value { font-size: 1.7rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.25; margin: 0.15rem 0; }
.kpi-note { font-size: 0.85rem; color: var(--muted); }
[class*="st-key-kpi_"] { height: 100%; background: #fff; border-radius: 14px; }
[data-testid="stExpander"] details { background: #fff; border-radius: 14px; }
[data-testid="stExpander"] summary { font-weight: 500; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; padding: 0.35rem 1rem; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { background: var(--ink); color: #f7f7f5;
                                                                          border: 0; border-radius: 12px; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { background: #333;
                                                                                           color: #fff; }
[data-testid="stPageLink"] a { border-radius: 10px; padding: 0.3rem 0.6rem; }
[data-testid="stPageLink"] a:hover { background: #ecebe7; }
.stAppDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

PAGES = {
    "home": st.Page(page_home, title="Start", icon=":material/home:", default=True),
    "pick": st.Page(page_pick, title="Evaluate options", url_path="compare"),
    "value": st.Page(page_value, title="Business case", url_path="value"),
    "ready": st.Page(page_ready, title="Readiness", url_path="readiness"),
    "case": st.Page(page_case, title="Decide", url_path="business-case"),
    "track": st.Page(page_track, title="Measure", url_path="track"),
    "pilot": st.Page(page_pilot, title="GenAI work orders", url_path="genai-pilot"),
    "agents": st.Page(page_agents, title="GenAI case review", url_path="agents"),
}
st.navigation({"": [PAGES["home"]],
               "Model": [PAGES[k] for k in ("pick", "value", "ready", "case", "track")],
               "GenAI tests": [PAGES["pilot"], PAGES["agents"]]}, position="top").run()
