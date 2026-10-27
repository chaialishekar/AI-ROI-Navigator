"""AI Use-Case ROI Navigator

An independent concept project: helps a company decide which AI use case to deploy,
prove its value in dollars, and check whether it is ready for production.

Overview          - what the project shows, headline results, and the steps to click into
Plan the project
  1 Match AI to client - match a provider's library of 12 AI solutions to one client's problem, data and risk
  2 Prove the value   - tune a real predictive maintenance model for dollars, not accuracy
  3 Check readiness   - pilot-to-production readiness check
  4 Business case     - one-page summary to share with finance
After launch
  5 Track the value   - forecast vs actual savings, adoption and drift (fed by the S3 -> Databricks pipeline)
Generative AI
  6 GenAI pilot       - measured Bedrock pilot of an AI work-order writer, and its ROI across five cost categories
  7 AI analyst agents - intake, memo and red-team agents on Bedrock, graded automatically (v1 vs v2 vs v3)

Every page follows one template: the question, the answer, three numbers, one chart and what to do next
(the client view), then 'See the analysis' with the assumptions, method and data, then Back / Next links.
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
from src.model import train_and_score
from src.matching import EXAMPLE_CLIENTS, INDUSTRIES, RISK_WEIGHT, URGENCY_WEIGHT, Client, match, problems, weights
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

st.set_page_config(page_title="AI Use-Case ROI Navigator", page_icon=":material/insights:", layout="wide")

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



def bar_layout(fig, height=300, money_axis=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2, title=""),
                      xaxis_title="", yaxis_title="")
    if money_axis:
        fig.update_layout(**{money_axis: dict(tickformat="$,.0f", title="")})
    return fig


# ---------------------------------------------------------------- Page 1
BLOCKER_FIX = {"data_readiness": "clean and connect the data before building",
               "ease_of_build": "start from a ready-made product rather than a custom build",
               "low_risk": "keep a person reviewing every output and start with one team",
               "speed_to_value": "agree an early milestone the client can see"}


def pick_client() -> Client:
    """Client chooser shown at the top of step 1: three example clients or your own."""
    options = list(EXAMPLE_CLIENTS) + ["Your own client"]
    name = st.segmented_control("Client", options, default=options[0], key="client_pick") or options[0]
    if name != "Your own client":
        c = EXAMPLE_CLIENTS[name]
        st.markdown(f'<div class="intro">{esc_html(c.blurb)}. Main problem: <b>{esc_html(c.problem.lower())}</b> · '
                    f'data maturity {c.data_maturity}/5 · {esc_html(c.risk_appetite.lower())} risk appetite · '
                    f'needs results {esc_html(c.urgency.lower())}.</div>', unsafe_allow_html=True)
        return c
    lib = load_use_cases()
    a, b, c3, d, e = st.columns([1.1, 1.5, 1, 1, 1.2])
    industry = a.selectbox("Industry", INDUSTRIES, key="cl_industry")
    problem = b.selectbox("Main problem", problems(lib, industry), key=f"cl_problem_{industry}")
    maturity = c3.slider("Data maturity", 1, 5, 3, key="cl_data", help="1 = scattered spreadsheets, 5 = clean, connected data")
    risk = d.selectbox("Risk appetite", list(RISK_WEIGHT), index=1, key="cl_risk")
    urgency = e.selectbox("Needs results", list(URGENCY_WEIGHT), key="cl_urgency")
    return Client("Your client", industry, problem, maturity, risk, urgency)


def page_pick():
    answer, client, detail = page_header(1, "Which AI fits this client?",
                                         "An AI provider's library of 12 solutions, matched to one client by their main "
                                         "problem, data, risk appetite and urgency. Pick a client.")
    with client:
        cl = pick_client()
    t_map, t_ranked, t_lib, t_how = detail.tabs(["Value vs feasibility map", "Ranked for this client",
                                                 "The whole library", "How matching works"])
    with t_lib:
        st.caption("The provider's library, scored 1–5 in general (5 is always better; for risk, 5 means low risk). "
                   "Edit any score; the match updates. Scores are illustrative.")
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
    who = "this client" if cl.name == "Your client" else f"the {cl.name.lower()}"
    blocker = min(FEASIBILITY_CRITERIA, key=lambda k: top[k])
    next_txt = (f"Fix the blocker first: {FEASIBILITY_CRITERIA[blocker].lower()} is {top[blocker]:.0f}/5, so "
                f"{BLOCKER_FIX[blocker]}." if top["quadrant"] == "Big bet" else NEXT_STEP[top["quadrant"]])
    answer.markdown(md(
        f"**{top['use_case']}** for {who}. {top['description']}. "
        + ("It solves their main problem and is a quick win." if top["quadrant"] == "Quick win" and top["solves_problem"]
           else f"It solves their main problem but is a {top['quadrant'].lower()}." if top["solves_problem"]
           else f"Nothing in the library targets {cl.problem.lower()} directly; this is the best fit.")))

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
            range_x=[1, 5.2], range_y=[1, 5.2], labels={"x_plot": "Feasibility for this client →", "y_plot": "Value to this client →"},
            height=420,
        )
        fig.add_vline(x=THRESHOLD, line_dash="dot", line_color="#8a979c")
        fig.add_hline(y=THRESHOLD, line_dash="dot", line_color="#8a979c")
        for x, y, label in [(4.6, 5.1, "QUICK WINS"), (2.2, 5.1, "BIG BETS"), (4.6, 1.15, "FILL-INS"), (2.2, 1.15, "DEPRIORITIZE")]:
            fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=11, color="#8a979c"))
        fig.update_traces(textposition="middle center", marker=dict(size=24), textfont=dict(color="white", size=11))
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("Numbers on the dots are the rank for this client.")
    with t_how:
        vw, fw = weights(cl)
        st.markdown(
            "1. **Industry:** only solutions that serve the client's industry are considered.\n"
            "2. **Main problem:** a solution that solves it gets full strategic fit and +1 financial impact; one that "
            "doesn't loses 2 points of strategic fit and 1 of financial impact.\n"
            "3. **Data maturity** shifts data readiness: 5/5 makes data work easier, 1/5 harder.\n"
            f"4. **Risk appetite** ({cl.risk_appetite.lower()}) sets the weight on 'low risk if the AI is wrong' to "
            f"{fw['low_risk']:.0%} of feasibility.\n"
            f"5. **Urgency** ({cl.urgency.lower()}) sets the weight on speed to value to {fw['speed_to_value']:.0%}.\n\n"
            "Then value × feasibility (each 1–5, so 1–25) ranks the rest. Quick win = both 3.5 or more. "
            "Code: `src/matching.py`.")

    colors = [QUADRANT_COLORS[q] for q in m["quadrant"]]
    figb = go.Figure(go.Bar(y=m["use_case"], x=m["priority"], orientation="h", marker=dict(color=colors, cornerradius=6),
                            text=[f"{p:.1f}" + ("  · solves their problem" if s else "") for p, s in zip(m["priority"], m["solves_problem"])],
                            textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f} / 25<extra></extra>"))
    figb.update_layout(height=60 + 42 * len(m), margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(range=[0, 30], title="Fit for this client (value × feasibility, out of 25)"),
                       yaxis=dict(autorange="reversed", title=""))

    with client:
        kpis([("Solutions that serve them", f"{len(m)} of {len(library)}", f"in {cl.industry.lower()}"),
              ("Solve their main problem", f"{fits}", cl.problem.lower()),
              ("Recommended", top["use_case"], f"{top['quadrant'].lower()} · fit {top['priority']:.1f} / 25")])
        st.markdown(f"**How each solution fits {who}** &nbsp; " + " &nbsp; ".join(
            f'<span style="color:{c}">●</span> {q.lower()}' for q, c in QUADRANT_COLORS.items()), unsafe_allow_html=True)
        st.plotly_chart(figb, width="stretch")
        next_steps([f"**{w}**" if i == 0 else w for i, w in enumerate(top["why"])]
                   + [f"**Next:** {next_txt}"], label="Why it fits")
    nav_footer("pick")


# ---------------------------------------------------------------- Page 2
def page_value():
    answer, client, detail = page_header(2, "Is it worth it?",
                                         "Worked example: the factory client and its match, predictive maintenance. "
                                         "A model trained on 10,000 machine records warns before a breakdown. "
                                         "Its catches, misses and false alarms are turned into dollars.")
    res = get_model_results()
    y_test = res["y_test"]
    t_costs, t_acc, t_risk, t_build = detail.tabs(["Your costs", "Accuracy vs value", "How sure are we?",
                                                   "Models and build options"])
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
        f"**Yes: {money(r['annual_net'])} a year, paying back in {months(r['payback_months'])}.** "
        + (f"Even if every cost estimate is too optimistic it still saves {money(low['annual_net'])} a year."
           if robust else f"But if the cost estimates are too optimistic it {'loses' if low['annual_net'] <= 0 else 'saves only'} "
           f"{money(abs(low['annual_net']))} a year, so firm them up first.")))

    figs = go.Figure(go.Bar(
        x=[f"{k} case" for k in cases], y=[c["annual_net"] for c in cases.values()],
        text=[money(c["annual_net"]) for c in cases.values()], textposition="outside",
        marker_color=["#a25f00", "#0f6d74", "#1d7a45"]))
    bar_layout(figs, 280, "yaxis")
    figs.update_yaxes(range=[min(0, low["annual_net"]) * 1.2, max(c["annual_net"] for c in cases.values()) * 1.2])

    with client:
        kpis([("Net savings a year", money(r["annual_net"]), f"90% range {money_range(band['annual_net'])}"),
              ("Payback", months(r["payback_months"]), f"{money(costs.deployment_cost)} to deploy"),
              ("3-year ROI", f"{r['roi_3y']*100:,.0f}%", f"first year {money(r['first_year'])}")])
        st.markdown("**Savings a year if our cost estimates are pessimistic, as planned, or optimistic**")
        st.plotly_chart(figs, width="stretch")
        next_steps([
            md(f"Agree the **{top['assumption'].replace(' ($)', '').lower()}** with finance: it is the biggest risk, "
               f"moving 3-year value by {money(top['swing'])}."),
            md(f"Set alerts for dollars, not accuracy (threshold {best_money['threshold']:.2f})"
               + (f": worth {money(gap)} more a year." if gap > 1 else ".")),
            md(f"Go live fast: every month of delay costs about {money(r['annual_net'] / 12)}."),
        ])
    nav_footer("value")


# ---------------------------------------------------------------- Page 3
EXAMPLE_PILOT = {"Data": [4, 4], "Workflow fit": [2, 2], "ROI clarity": [4, 3],
                 "User trust": [3, 2], "Governance & risk": [3, 3]}


def page_ready():
    answer, client, detail = page_header(3, "Are we ready to launch?",
                                         "Most AI pilots stall on people and process, not the model. "
                                         "Ten statements, five areas, pre-filled for an example plant.")
    t_rate, t_rules = detail.tabs(["Rate your team", "How the verdict works"])
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
              ("Average", f"{avg:.1f} / 5", "3.5 or more to go"),
              ("Weakest area", weakest[0][0], f"{weakest[0][1]:.1f} / 5")])
        st.plotly_chart(figr, width="stretch")
        next_steps([f"**{d}** ({s:.1f}): {ACTIONS[d]}" for d, s in weakest])
    nav_footer("ready")


# ---------------------------------------------------------------- Page 4
def page_case():
    answer, client, detail = page_header(4, "The business case",
                                         "Steps 1–3 and the value tracked after launch, on one page for finance. "
                                         "It updates when you change inputs on the other pages.",
                                         detail_label="Preview the one-pager")
    val = get_value()
    rd = get_readiness()
    tk = st.session_state.get("tracking") or default_tracking()
    answer.markdown(md(
        f"**Worth doing: {money(val['annual_net'])} a year, {months(val['payback_months'])} payback.** Readiness says "
        f"**{rd['verdict'].lower()}**" + (f", so fix {rd['weakest'][0][0].lower()} before launch." if rd["verdict"] != "Go" else ".")))
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
""" + "".join(f"- Fix: {d} ({s:.1f}): {ACTIONS[d]}\n" for d, s in rd.get("weakest", [])) + tracking_md + """
## Key assumptions
- Plant costs as entered on the Prove the value page (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
    with client:
        kpis([("Net savings a year", money(val["annual_net"]), f"payback {months(val['payback_months'])}"),
              ("Readiness", rd["verdict"], f"average {rd['avg']:.1f} / 5"),
              ("Delivered after launch", f"{tk['pct']*100:.0f}%" if tk and np.isfinite(tk["pct"]) else "n/a",
               f"of forecast, {tk['months']} months" if tk else "")])
        st.download_button("Download the one-pager (.md)", summary, file_name="ai_business_case.md", key="dl_summary",
                           type="primary")
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


def monthly_vs_forecast(t: pd.DataFrame, ok_color: str, alert_color: str, height: int = 320) -> go.Figure:
    """Actual net savings per month as bars, the monthly forecast as a dashed line; alert months highlighted."""
    alert = t["adoption_alert"] | t["drift_alert"]
    why = np.where(t["drift_alert"] & t["adoption_alert"], "adoption and drift alert",
                   np.where(t["drift_alert"], "drift alert", np.where(t["adoption_alert"], "adoption alert", "on track")))
    fig = go.Figure([
        go.Bar(x=t["month"], y=t["actual"], name="Actual", showlegend=False, marker=dict(color=np.where(alert, alert_color, ok_color),
                                                                         cornerradius=6),
               customdata=why, hovertemplate="%{x}: %{y:$,.0f} (%{customdata})<extra></extra>"),
        go.Scatter(x=t["month"], y=t["forecast"], name="Forecast", mode="lines", line=dict(color="#15222a", dash="dash", width=2),
                   hovertemplate="Forecast %{y:$,.0f}<extra></extra>")])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(tickformat="$,.0f", title=""),
                      xaxis_title="", legend=dict(orientation="h", y=-0.2), bargap=0.25)
    return fig


def page_track():
    answer, client, detail = page_header(5, "Is it paying off?",
                                         "After launch, monthly counts from the maintenance system are turned into "
                                         "dollars and compared with the forecast.")
    val = get_value()
    sample = pd.read_csv("data/sample_actuals.csv")
    pipe = load_pipeline()
    t_src, t_health, t_gap, t_data = detail.tabs(["Data source", "Adoption and drift", "Why the gap", "Monthly data"])
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
                f"**Pipeline:** monthly files land in an encrypted S3 bucket (us-east-2), are synced to Databricks and "
                f"flow through bronze (as received) → silver (checked) → gold (dollars, adoption, drift). "
                f"Last refresh {log['refreshed_at'][:16].replace('T', ' ')} UTC: {log['bronze_rows']} files received, "
                f"{log['silver_rows']} passed the checks, {len(log['quarantined'])} quarantined. "
                + ("Gold matches this page's formula exactly." if pipe["matches"] else "Warning: gold differs from this page's formula.")))
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
        f"**Partly: {pct*100:.0f}% of the forecast** after {n} months ({money(delivered)} of {money(promised)}). "
        + (f"The biggest shortfall: {SHORTFALL[tk['worst_cause']]}. " if tk["worst_cause"] else "")
        + ("And the model is now catching fewer failures." if tk["drift_now"] else "")))

    with t_health:
        st.markdown(
            (f"**Adoption:** the crew acted on {last['adoption']*100:.0f}% of alerts last month "
             f"(target {ADOPTION_FLOOR:.0%})" + (f"; it was below target in {', '.join(low_months)}." if low_months else "."))
            + "  \n"
            + (f"**Drift:** recall over the last {DRIFT_WINDOW} months is {recall_now*100:.0f}% vs {val['recall']*100:.0f}% "
               f"at launch; alert when {DRIFT_DROP*100:.0f}+ points lower"
               + (f" (flagged in {', '.join(drift_months)})." if drift_months else ".")))
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
        steps.append(f"**Retrain the model** on recent data: it now catches {recall_now*100:.0f}% of failures, "
                     f"down from {val['recall']*100:.0f}% at launch.")
    if tk["worst_cause"] and not (tk["worst_cause"] == "Model recall" and tk["drift_now"]):
        cause = SHORTFALL[tk["worst_cause"]]
        steps.append(md(f"**{cause[0].upper() + cause[1:]}** ({money(tk['worst_effect'])}): {GAP_NEXT[tk['worst_cause']]}"))
    if last["adoption_alert"]:
        steps.append(f"**Win back the crew:** only {last['adoption']*100:.0f}% of alerts were acted on last month.")
    for q in (log or {}).get("quarantined", []):
        steps.append(f"**Ask the plant to resend the {q['month']} file**: it failed the data checks (details under the analysis).")

    with client:
        kpis([(f"Savings in {n} months", money(delivered), f"{money(delivered - promised)} vs forecast"),
              ("Forecast delivered", f"{pct*100:.0f}%" if np.isfinite(pct) else "n/a", f"of {money(promised)}"),
              ("Failures caught now", f"{recall_now*100:.0f}%" if np.isfinite(recall_now) else "n/a",
               f"{val['recall']*100:.0f}% at launch")])
        st.markdown("**Net savings each month vs the forecast** (amber = a month with an adoption or drift alert)")
        st.plotly_chart(figc, width="stretch")
        if steps:
            next_steps(steps)
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
    answer, client, detail = page_header(6, "Does GenAI pay for itself?",
                                         "When the model raises an alert, a small language model on Amazon Bedrock "
                                         "drafts the work order. We ran 30 real alerts through it, twice.")
    pilot = load_pilot()
    s1, s2 = pilot["summary"][1], pilot["summary"][2]
    t_res, t_roi, t_drafts = detail.tabs(["Pilot results", "ROI calculator", "Compare drafts"])
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
        f"**Yes, if the AI only writes and code does the diagnosis.** Letting the AI diagnose too gave the right cause "
        f"just {s1['correct_on_real_failures']:.0%} of the time; with a simple rule check it was "
        f"{s2['correct_on_real_failures']:.0%}. Review time, not the model fee, decides the ROI."))

    metrics = ["All 30 alerts", "24 real failures", "6 false alarms"]
    figv = go.Figure([go.Bar(name=name, x=metrics, y=[s["correct_mode_rate"] * 100, s["correct_on_real_failures"] * 100,
                                                       s["false_alarm_handled_rate"] * 100],
                             text=[f"{x:.0%}" for x in (s["correct_mode_rate"], s["correct_on_real_failures"],
                                                         s["false_alarm_handled_rate"])],
                             textposition="outside", marker_color=color)
                      for name, s, color in (("Version 1: AI diagnoses", s1, "#8a979c"),
                                             ("Version 2: code diagnoses, AI writes", s2, "#0f6d74"))])
    bar_layout(figv, 280)
    figv.update_yaxes(range=[0, 115], ticksuffix="%")

    with client:
        kpis([("Right cause, real failures", f"{s1['correct_on_real_failures']:.0%} → {s2['correct_on_real_failures']:.0%}",
               "version 1 → version 2"),
              ("Model cost", f"${s2['cost_per_task']*1000:.2f}", "per 1,000 drafts"),
              ("Net value a year", money(r2["net"]), f"ROI {r2['roi']*100:,.0f}%, version 2")])
        st.markdown("**Drafts that named the right cause**")
        st.plotly_chart(figv, width="stretch")
        next_steps([
            "**Keep the diagnosis in code**; let the AI write the work order.",
            "**Keep a human review**: 4 of 30 drafts still mentioned computer parts.",
            f"**Measure review time** in a wider pilot: above {r2['break_even_review_minutes']:.0f} minutes a draft, "
            "the assistant loses money.",
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
                                         "Three small agents on Amazon Bedrock: **Intake** turns a manager's paragraph "
                                         "into calculator inputs, **Memo** writes the CFO memo, **Red team** plays the "
                                         "skeptical CFO. Each is graded on cases with known answers.")
    AGENTS = load_agents()
    sc = AGENTS["scores"]
    cases = {k: {c["id"]: c for c in v} for k, v in AGENTS["cases"].items()}
    n_memo = len(cases["memo"])
    answer.markdown(md(
        f"**The routine parts, yes, once code checks the numbers and the AI only writes.** Intake went from "
        f"{sc['v1']['intake_field_accuracy']:.0%} to {sc['v3']['intake_field_accuracy']:.0%} of fields right and the red "
        f"team from {sc['v1']['redteam_caught']} to {sc['v3']['redteam_caught']} of {sc['v3']['redteam_flaws']} planted "
        "flaws caught. Deciding what to do about them stays with people."))

    versions = {"v1": "v1: model does everything", "v2": "v2: code does the numbers", "v3": "v3: code checklist judges"}
    rows = {"Intake fields right": lambda s: s["intake_field_accuracy"],
            "Memos with no invented numbers": lambda s: s["memo_faithful"] / n_memo,
            "Planted flaws caught": lambda s: s["redteam_caught"] / s["redteam_flaws"]}
    figv = go.Figure([go.Bar(name=label, x=list(rows), y=[f(sc[v]) * 100 for f in rows.values()],
                             text=[f"{f(sc[v]):.0%}" for f in rows.values()], textposition="outside", marker_color=color)
                      for (v, label), color in zip(versions.items(), ("#c4ccd0", "#8a979c", "#0f6d74"))])
    bar_layout(figv, 280)
    figv.update_yaxes(range=[0, 115], ticksuffix="%")

    with client:
        kpis([("Intake fields right", f"{sc['v1']['intake_field_accuracy']:.0%} → {sc['v3']['intake_field_accuracy']:.0%}",
               "version 1 → version 3"),
              ("Planted flaws caught", f"{sc['v3']['redteam_caught']} of {sc['v3']['redteam_flaws']}",
               f"{sc['v3']['redteam_clean_high']} false alarms on the sound case"),
              ("Cost of the whole test", f"${sc['v3']['total_cost']:.4f}", "Amazon Nova Lite")])
        st.markdown("**Score by version**")
        st.plotly_chart(figv, width="stretch")
        next_steps([
            "**Code does numbers and checks, the model does language.** Every fix moved a judgement out of the prompt "
            "and into code.",
            "**Grade agents on cases with known answers before trusting them.** Version 1 read well and was wrong "
            "most of the time.",
            "**Keep a person on the decision.** The checklist was written knowing the flaws, so it is not a blind test.",
        ], label="What this means for a team")

    t_sc, t_in, t_memo, t_rt, t_live = detail.tabs(["Scorecard", "Intake", "Memo", "Red team", "Try it live"])
    with t_sc:
        col = lambda v: [
            f"{sc[v]['intake_field_accuracy']:.0%}", f"{sc[v]['intake_invented_fields']}",
            f"{sc[v]['memo_faithful']} of {n_memo}", f"{sc[v]['redteam_caught']} of {sc[v]['redteam_flaws']}",
            f"{sc[v]['redteam_caught_first']} of {sc[v]['redteam_flaws']}", f"{sc[v]['redteam_clean_high']}",
            f"${sc[v]['total_cost']:.4f}"]
        st.dataframe(pd.DataFrame({label: col(v) for v, label in versions.items()},
                                  index=["Intake: fields right (8 descriptions)", "Intake: numbers invented when not stated",
                                         "Memo: memos with no invented numbers", "Red team: planted flaw raised as high/medium",
                                         "Red team: planted flaw listed first", "Red team: false 'high' alarms on the sound case",
                                         "Cost of the whole eval run"]), width="stretch")
        st.markdown(md(
            "**What changed.** v1 wrote arithmetic like `300 * 52` inside its JSON, invented a $5,000 figure in a memo, and gave "
            "nearly the same five objections to every case. **v2:** the model only quotes the numbers it was given; code "
            "converts weeks to years, checks each quote is really in the text, and fills memo placeholders like "
            "`{First-year ROI}`. Intake and memo became exact, but the red team still called \"15 of 20 minutes saved\" "
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
SECTIONS = {1: "Plan the project", 2: "Plan the project", 3: "Plan the project", 4: "Plan the project",
            5: "After launch", 6: "Generative AI", 7: "Generative AI"}
ORDER = ["home", "pick", "value", "ready", "case", "track", "pilot", "agents"]


def esc_html(text: str) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def page_header(step: int, title: str, intro: str, detail_label: str = "See the analysis: assumptions, method and data"):
    """Every page: question, one line of context, the answer, the client view, then the analysis on demand.

    Returns (answer, client, detail) containers; the page fills them in any order, so inputs inside the
    detail tabs can be read before the answer and the client view are written."""
    st.markdown(f'<div class="eyebrow">Step {step} of 7 · {SECTIONS[step]}</div>', unsafe_allow_html=True)
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
    with st.container(key=f"next_{label.split()[0].lower()}"):
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
WHY_URL = ""   # link to the Substack post; the overview shows the title without a link until this is set


@st.cache_data
def accuracy_vs_profit() -> dict:
    """Savings at the most accurate vs the most profitable alert threshold, default plant costs."""
    res = get_model_results()
    curve = sweep(res["y_test"], res["models"]["Gradient boosting"]["proba"], Costs())
    return {"accurate": float(curve.loc[curve["accuracy"].idxmax(), "annual_net"]),
            "profitable": float(curve.loc[curve["annual_net"].idxmax(), "annual_net"])}


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


def tour_content(step: str) -> dict:
    """Question, plain answer, chips and a small chart for one step of the overview tour."""
    val, rd = get_value(), get_readiness()
    if step == "Choose":
        lib = load_use_cases()
        tops = {name: match(lib, c).iloc[0] for name, c in EXAMPLE_CLIENTS.items()}
        m = match(lib, EXAMPLE_CLIENTS["Mid-size factory"])
        return dict(page="pick", label="Step 1 · Match AI to client", q="Which AI from the library fits this client?",
                    a="A provider's library of 12 AI solutions across manufacturing, retail and healthcare. Each client is "
                      "matched by their main problem, how good their data is, how much risk they accept and how fast "
                      "they need results. Same library, different clients, different answers:",
                    chips=[f"{name} → {r['use_case']}" for name, r in tops.items()],
                    chart_title="Fit for the factory client, out of 25 (lime = solves their main problem)",
                    fig=mini_bars(m["use_case"], m["priority"], [f"{p:.1f}" for p in m["priority"]],
                                  m["solves_problem"], horizontal=True))
    if step == "Prove":
        ap = accuracy_vs_profit()
        return dict(page="value", label="Step 2 · Prove the value", q="What is predictive maintenance worth in dollars?",
                    a=f"A model trained on 10,000 real machine records warns before a breakdown. Tuned for savings "
                      f"instead of accuracy, it saves {money(val['annual_net'])} a year and pays back in "
                      f"{months(val['payback_months'])}: {money(ap['profitable'] - ap['accurate'])} a year more than "
                      "the most accurate setting.",
                    chips=["Real dataset: AI4I 2020", "3 models compared", "Low / base / high case"],
                    chart_title="Net savings a year, same model, two alert settings",
                    fig=mini_bars(["Most accurate", "Most profitable"], [ap["accurate"], ap["profitable"]],
                                  [money(ap["accurate"]), money(ap["profitable"])], [False, True]))
    if step == "Get ready":
        dims = rd.get("dims") or {d: sum(v) / len(v) for d, v in EXAMPLE_PILOT.items()}
        weak = {d for d, _ in rd["weakest"][:2]}
        return dict(page="ready", label="Step 3 · Check readiness", q="Is the team ready to launch it?",
                    a=f"Most AI pilots stall on people and process, not the model. A 10-question check across 5 areas "
                      f"says **{rd['verdict'].lower()}**: the model and data are fine, but "
                      f"{' and '.join(d.lower() for d, _ in rd['weakest'][:2])} need work first.",
                    chips=[f"Fix: {d}" for d, _ in rd["weakest"][:2]] + ["Then: one-page business case"],
                    chart_title="Readiness score out of 5 (lime = fix first)",
                    fig=mini_bars(list(dims), list(dims.values()), [f"{v:.1f}" for v in dims.values()],
                                  [d in weak for d in dims], horizontal=True))
    if step == "Keep it paying":
        t = default_track_frame()
        tk = default_tracking()
        drift = t.loc[t["drift_alert"], "month"].tolist()
        fig = mini_chart(monthly_vs_forecast(t, SAGE, "#e3a948", height=260), 260)
        fig.update_yaxes(showticklabels=False)
        return dict(page="track", label="Step 5 · Track the value", q="After launch, is it delivering what was promised?",
                    a=f"Monthly maintenance data flows from S3 through a Databricks pipeline. After {tk['months']} months "
                      f"it delivered **{tk['pct']*100:.0f}% of the forecast**. The app splits the gap into causes and "
                      + (f"flagged in {drift[0]} that the model had started missing failures." if drift
                         else "watches for the model getting worse."),
                    chips=["S3 → Databricks", "Bronze · silver · gold", "Drift and adoption alerts"],
                    chart_title="Net savings each month vs forecast (amber = alert month, simulated rollout)", fig=fig)
    pilot = load_pilot()["summary"]
    if step == "GenAI pilot":
        s1, s2 = pilot[1], pilot[2]
        return dict(page="pilot", label="Step 6 · GenAI pilot", q="Does a GenAI assistant pay for itself?",
                    a=f"A small language model on Amazon Bedrock drafts the repair work order for each alert. When it "
                      f"also diagnosed the fault it named the right cause only {s1['correct_on_real_failures']:.0%} "
                      f"of the time; with code doing the diagnosis, {s2['correct_on_real_failures']:.0%}. "
                      f"The model costs about ${s2['cost_per_task']*1000:.2f} per 1,000 drafts: review time decides the ROI.",
                    chips=["30 real alerts", "Amazon Nova Micro", "5 cost categories"],
                    chart_title="Drafts with the right cause, 24 real failures",
                    fig=mini_bars(["AI diagnoses", "Code diagnoses, AI writes"],
                                  [s1["correct_on_real_failures"], s2["correct_on_real_failures"]],
                                  [f"{s1['correct_on_real_failures']:.0%}", f"{s2['correct_on_real_failures']:.0%}"],
                                  [False, True]))
    sc = load_agents()["scores"]
    caught = [sc[v]["redteam_caught"] for v in ("v1", "v2", "v3")]
    return dict(page="agents", label="Step 7 · AI analyst agents", q="Can AI agents do the analyst's paperwork?",
                a=f"Three agents turn a manager's paragraph into inputs, write the CFO memo and play the skeptical CFO. "
                  f"Graded on cases with known answers, the red team went from {caught[0]} to {caught[2]} of "
                  f"{sc['v3']['redteam_flaws']} planted flaws caught once code checked the numbers, for about "
                  f"${sc['v3']['total_cost']:.3f} a run.",
                chips=["Intake · Memo · Red team", "19 graded cases", "Amazon Nova Lite"],
                chart_title=f"Planted flaws caught, out of {sc['v3']['redteam_flaws']}",
                fig=mini_bars(["Version 1", "Version 2", "Version 3"], caught, [str(c) for c in caught],
                              [False, False, True]))


def page_home():
    val, tk = get_value(), default_tracking()
    c = val["costs"]
    lost = c.failures_per_year * c.cost_missed
    with st.container(key="hero"):
        left, right = st.columns([1.3, 1], gap="large")
        left.markdown(
            '<div class="eyebrow">AI value engineering · portfolio project</div>'
            '<div class="hero-title">Which AI fits this client, and is it still paying off?</div>'
            '<div class="hero-text">An AI provider has a library of 12 solutions. For every client, a value engineer has '
            'to say which one fits, what it is worth to them, whether they are ready, and, after launch, whether it '
            'delivers. Most AI pilots never prove their worth; this app answers each question with numbers, worked '
            'end to end for one client:</div>'
            + "".join(f'<span class="chip">{esc_html(x)}</span>' for x in
                      ("A mid-size factory", f"{c.failures_per_year} surprise breakdowns a year",
                       f"{money(c.cost_missed)} each", "AI that warns before a machine fails")),
            unsafe_allow_html=True)
        right.markdown(
            f'<div class="impact"><div class="impact-label">Today</div><div class="impact-num">{esc_html(money(lost))}</div>'
            '<div class="impact-note">lost a year to surprise breakdowns</div></div>'
            f'<div class="impact"><div class="impact-label">With AI</div><div class="impact-num"><span class="lime">'
            f'{esc_html(money(val["annual_net"]))}</span></div><div class="impact-note">saved a year after AI costs · '
            f'pays back in {esc_html(months(val["payback_months"]))}</div></div>'
            f'<div class="impact"><div class="impact-label">{tk["months"]} months after launch</div>'
            f'<div class="impact-num">{tk["pct"]*100:.0f}%</div><div class="impact-note">of the promised savings '
            'delivered, with the reasons for the gap</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="row-label">Explore the journey · pick a step</div>', unsafe_allow_html=True)
    step = st.segmented_control("Step", ["Choose", "Prove", "Get ready", "Keep it paying", "GenAI pilot", "AI agents"],
                                default="Prove", key="tour", label_visibility="collapsed") or "Prove"
    tc = tour_content(step)
    with st.container(key="tour_panel"):
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            st.markdown(f'<div class="eyebrow">{esc_html(tc["label"])}</div>'
                        f'<div class="tour-q">{esc_html(tc["q"])}</div>', unsafe_allow_html=True)
            st.markdown(md(tc["a"]))
            st.markdown("".join(f'<span class="chip soft">{esc_html(x)}</span>' for x in tc["chips"]),
                        unsafe_allow_html=True)
            st.page_link(PAGES[tc["page"]], label=f"Open {tc['label'].replace(' · ', ': ', 1).replace('Step', 'step')}",
                         icon=":material/arrow_forward:")
        with right:
            st.markdown(f'<div class="chart-title">{esc_html(tc["chart_title"])}</div>', unsafe_allow_html=True)
            st.plotly_chart(tc["fig"], width="stretch", config={"displayModeBar": False, "staticPlot": True})

    st.markdown('<div class="row-label">What changes for the business</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    for col, (icon, head, body) in zip((i1, i2, i3), (
            ("01", "The right AI for each client", "Matched to the client's problem, data and risk appetite, not the "
                                                   "provider's favourite product."),
            ("02", "A number finance can sign", f"{money(val['annual_net'])} a year, with a pessimistic case and the "
                                               "one assumption to check first."),
            ("03", "Problems caught in months, not years", "Ignored alerts and a weakening model show up in the "
                                                         "monthly data, with what to do about each."))):
        col.markdown(f'<div class="why"><div class="why-icon">{icon}</div><b>{esc_html(head)}</b>'
                     f'<div class="why-text">{esc_html(body)}</div></div>', unsafe_allow_html=True)

    post = "AI's Hidden Problem: Nobody Revisits the Original Decision"
    st.markdown(f'<div class="why-built"><b>Why I built this.</b> In <i>{"<a href=" + chr(34) + WHY_URL + chr(34) + ">" + post + "</a>" if WHY_URL else post}</i> '
                "I argued that most AI waste comes from decisions that were right when they were made and never revisited "
                "once the facts changed. Step 5 is that revisit: it goes back to the business case every month.</div>",
                unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1.expander("What is real, what is illustrative"):
        st.markdown("**Real:** the AI4I 2020 machine dataset and the model trained on it, the Amazon Bedrock pilot and "
                    "agent runs, the S3 → Databricks pipeline.  \n**Illustrative, editable:** plant costs, use-case "
                    "scores and the 12-month rollout data. Independent concept project.")
    with e2.expander("How it is built"):
        st.markdown("Python and scikit-learn · Amazon Bedrock (Nova) · S3 and Databricks (bronze → silver → gold) · "
                    "Streamlit. [Code on GitHub](https://github.com/chaialishekar/AI-ROI-Navigator).")
    with e3.expander("What comes next"):
        st.markdown("Blind tests for the agents, a monthly scheduled pipeline with alerts, and any AI use case (not just "
                    "maintenance). [See the roadmap](https://github.com/chaialishekar/AI-ROI-Navigator/blob/main/docs/roadmap.md).")
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
.impact { background: rgba(255, 255, 255, 0.72); border-radius: 22px; padding: 0.8rem 1.2rem; margin-bottom: 0.6rem;
          transition: transform 0.25s ease, box-shadow 0.25s ease; }
.impact:hover { transform: translateX(-4px); box-shadow: 0 8px 24px rgba(29, 37, 33, 0.08); }
.impact-label { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; font-weight: 700; color: #6b766d; }
.impact-num { font-size: 2.5rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1.15; color: #1d2521; }
.impact-num .lime { background: #d4ee5e; border-radius: 14px; padding: 0 0.45rem; }
.impact-note { font-size: 0.88rem; color: #4e5952; }
.st-key-tour_panel { background: #fff; border-radius: 28px; padding: 1.6rem 1.9rem; box-shadow: 0 12px 36px rgba(29, 37, 33, 0.07); }
.tour-q { font-size: 1.7rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.01em; margin: 0.3rem 0 0.6rem; }
.chart-title { font-size: 0.85rem; color: #6b766d; margin-bottom: 0.2rem; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; padding: 0.35rem 1rem; }
.why { background: #fff; border-radius: 22px; padding: 1.1rem 1.3rem; min-height: 9.5rem; margin-bottom: 0.4rem; box-shadow: 0 4px 16px rgba(29, 37, 33, 0.05); }
.why-icon { display: table; font-size: 0.78rem; font-weight: 700; background: #d4ee5e; border-radius: 999px; padding: 0.15rem 0.6rem; margin-bottom: 0.5rem; }
.why-built { margin: 0.9rem 0 0.4rem; padding: 0.9rem 1.2rem; border-radius: 18px; background: #eef4d6; font-size: 0.95rem; }
.why-built a { color: #1d2521; }
.why-text { font-size: 0.9rem; color: #4e5952; margin-top: 0.2rem; }
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
    "pick": st.Page(page_pick, title="1 · Match AI to client", url_path="match"),
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
