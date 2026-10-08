"""AI Use-Case ROI Navigator

An independent concept project: helps a company decide which AI use case to deploy,
prove its value in dollars, and check whether it is ready for production.

Tab 1  Prioritize   - score use cases on value vs feasibility
Tab 2  Prove value  - tune a real predictive maintenance model for dollars, not accuracy
Tab 3  Readiness    - pilot-to-production readiness check
Tab 4  Business case - one-page summary to share with finance
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
from src.value import Costs, evaluate, plain_language, sweep

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

tab1, tab2, tab3, tab4 = st.tabs(["1 · Prioritize", "2 · Prove value", "3 · Readiness", "4 · Business case"])


@st.cache_resource
def get_model_results():
    return train_and_score()


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x/1e6:.2f}M" if x >= 1e6 else f"{sign}${x/1e3:,.0f}k"


def md(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't render dollar amounts as math."""
    return text.replace("$", "\\$")


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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annual net savings", money(r["annual_net"]))
    m2.metric("Payback", f"{r['payback_months']:.1f} months" if np.isfinite(r["payback_months"]) else "Never")
    m3.metric("3-year ROI", f"{r['roi_3y']*100:,.0f}%")
    m4.metric("First-year value", money(r["first_year"]), help="Savings while live in year one, minus deployment cost")
    st.info(f"**In plain language:** {plain_language(r)}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Failures caught / year", f"{r['caught']:.0f}")
    k2.metric("Failures missed / year", f"{r['missed']:.0f}")
    k3.metric("False alarms / year", f"{r['false_alarms']:.0f}")
    k4.metric("Accuracy", f"{r['accuracy']*100:.1f}%", help="Share of all test readings classified correctly")

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

    st.caption(
        f"Model trained on {res['n_train']:,} records and tested on {res['n_test']:,} records it never saw "
        f"({int(y_test.sum())} real failures). Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0). "
        "Dollar results scale the test-set rates to your plant's failures per year."
    )
    st.session_state["value"] = {**r, "model": model_name}

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

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.subheader("Business case summary")
    val = st.session_state.get("value", {})
    rd = st.session_state.get("readiness", {})
    if not val:
        st.info("Open the Prove value tab first.")
    else:
        summary = f"""# AI Business Case: Predictive Maintenance

**Use case:** Predictive maintenance (manufacturing). Top-ranked use case in the Prioritize tab: {st.session_state.get('top_use_case', 'n/a')} ({st.session_state.get('top_quadrant', 'n/a')}).

## Value
- Annual net savings: {money(val['annual_net'])}
- Payback: {val['payback_months']:.1f} months
- 3-year ROI: {val['roi_3y']*100:,.0f}%
- First-year value (after deployment cost): {money(val['first_year'])}
- Model: {val['model']}, alert threshold {val['threshold']:.2f}
- {plain_language(val)}
- Per year: ~{val['caught']:.0f} failures caught, ~{val['missed']:.0f} missed, ~{val['false_alarms']:.0f} false alarms

## Readiness
- Verdict: {rd.get('verdict', 'n/a')} (average {rd.get('avg', 0):.1f} / 5)
""" + "".join(f"- Fix: {d} ({s:.1f}): {ACTIONS[d]}\n" for d, s in rd.get("weakest", [])) + """
## Key assumptions
- Plant costs as entered in the Prove value tab (illustrative defaults otherwise)
- Model results from held-out test data (AI4I 2020 dataset), scaled to the plant's failures per year
"""
        st.markdown(md(summary))
        st.download_button("Download summary (.md)", summary, file_name="ai_business_case.md", key="dl_summary")
