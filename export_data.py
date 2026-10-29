"""Export the case-study numbers for the static site (site/assets/data.js).

Every figure the site shows comes from the same src/ functions the Streamlit app uses, so the two cannot drift.
Re-run after changing the model, the library or the plant's costs:

    python site/export_data.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.library import EXAMPLE_CLIENTS, evidence_strength, load_library, rank  # noqa: E402
from src.model import load_data, train_and_score  # noqa: E402
from src.readiness import ACTIONS, verdict  # noqa: E402
from src.tracking import COLUMNS, explain_gap, track  # noqa: E402
from src.value import ACCURACY_THRESHOLD, Costs, alert_threshold, default_ranges, evaluate, payback, scenarios  # noqa: E402
from src.workorder import rule_alerts  # noqa: E402

# Same example readiness ratings as the app's Readiness page (two raters per dimension)
EXAMPLE_PILOT = {"Data": [4, 4], "Fits how people work": [2, 2], "Clear payoff": [4, 3],
                 "People trust it": [3, 2], "Clear ownership": [3, 3]}
LEAK_SHORT = {"Unplanned machine downtime": "Downtime", "Scrap, rework and warranty defects": "Scrap and rework",
              "Shortages, expediting and excess inventory": "Shortages"}
CHARACTER = {"Predictive maintenance": "sentry", "Visual defect inspection": "lens",
             "Demand and supply forecasting": "foresight"}


def num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def library_options(client) -> list[dict]:
    expected, cautious = rank(load_library(), client), rank(load_library(), client, cautious=True)
    out = []
    for _, r in expected.iterrows():
        c = cautious[cautious["solution"] == r["solution"]].iloc[0]
        out.append({
            "key": CHARACTER[r["solution"]], "solution": r["solution"], "does": r["what_it_does"],
            "problem": LEAK_SHORT.get(r["problem"], r["problem"]), "problem_cost": num(r["problem_cost"]),
            "rank": int(r["rank"]), "rank_cautious": int(c["rank"]),
            "net3": num(r["net3_mid"]), "net3_cautious": num(c["net3_low"]),
            "payback": num(r["payback"]), "payback_cautious": num(c["payback"]),
            "improve_low": num(r["improvement_low"]), "improve_high": num(r["improvement_high"]),
            "setup": num(r["setup"]), "running": num(r["running"]), "months": num(r["months"]),
            "evidence": r["evidence"], "strength": evidence_strength(r["evidence"]),
            "source": r["source"], "source_url": r["source_url"]})
    return out


def model_value(res: dict, costs: Costs) -> dict:
    """The business-case result: gradient boosting at the break-even alert level (as in the app)."""
    y, proba = res["y_test"], res["models"]["Gradient boosting"]["proba"]
    thr = alert_threshold(costs)
    r = evaluate(y, proba, thr, costs)
    cases = scenarios(y, proba, thr, costs, default_ranges(costs))
    return {**r, "cases": cases, "accurate_net": evaluate(y, proba, ACCURACY_THRESHOLD, costs)["annual_net"]}


def decision_rule(val: dict, rd: dict) -> tuple[str, str]:
    """Mirrors decision_rule() in app.py."""
    worst = val["cases"]["Low"]["annual_net"]
    if val["annual_net"] <= 0:
        return "Stop", "Even as planned it costs more than it saves."
    if worst <= 0:
        return "Revise", "Firm up the cost estimates first: it loses money if they are too optimistic."
    if rd["verdict"] == "Not ready":
        return "Revise", "The team is not ready: close the gaps and check again."
    if rd["verdict"] == "Fix first":
        return "Proceed after fixes", "Close the readiness gaps first. It saves money even in the worst case."
    return "Proceed", "It saves money even in the worst case and the team is ready."


def main():
    costs = Costs()
    client = EXAMPLE_CLIENTS["Auto-parts plant"]
    res = train_and_score()
    val = model_value(res, costs)
    rules = evaluate(res["y_test"], rule_alerts(), 0.5, costs)

    lib = load_library().query("solution == 'Predictive maintenance' and industry == 'Automotive'").iloc[0]
    downtime = costs.failures_per_year * costs.cost_missed
    published = downtime * (lib["improvement_low"] + lib["improvement_high"]) / 2 - costs.running_cost

    dims = {d: sum(v) / len(v) for d, v in EXAMPLE_PILOT.items()}
    v, weakest = verdict(dims)
    rd = {"verdict": v, "avg": sum(dims.values()) / len(dims), "weakest": weakest}
    label, reason = decision_rule(val, rd)

    pipe_path = "data/pipeline/tracked_value.csv"
    actuals, source = ((pd.read_csv(pipe_path)[COLUMNS], "S3 → Databricks pipeline") if os.path.exists(pipe_path)
                       else (pd.read_csv("data/sample_actuals.csv"), "simulated sample"))
    t = track(actuals, val, costs)
    gap = explain_gap(t, val, costs)

    df = load_data()
    causes = {"TWF": "Tool wear", "HDF": "Heat", "PWF": "Power", "OSF": "Overstrain", "RNF": "Random"}

    case = {
        "plant": {"name": client.name, "blurb": client.blurb, "total": sum(client.problem_costs.values()),
                  "losses": [{"name": LEAK_SHORT[k], "cost": v} for k, v in client.problem_costs.items()]},
        "options": library_options(client),
        "costs": {k: num(getattr(costs, k)) for k in costs.__dataclass_fields__},
        "value": {k: num(val[k]) for k in ("annual_net", "threshold", "recall", "precision", "accuracy",
                                           "caught", "missed", "false_alarms", "roi_3y", "accurate_net")}
                 | {"payback": num(payback(val["annual_net"], costs)),
                    "cases": {k: num(c["annual_net"]) for k, c in val["cases"].items()}},
        "published": {"net": num(published), "low_pct": num(lib["improvement_low"]),
                      "high_pct": num(lib["improvement_high"])},
        "rules": {k: num(rules[k]) for k in ("annual_net", "recall", "precision", "false_alarms")},
        "readiness": {"verdict": v, "avg": num(rd["avg"]), "dims": {d: num(s) for d, s in dims.items()},
                      "weakest": [{"name": d, "score": num(s), "action": ACTIONS[d]} for d, s in weakest[:2]]},
        "decision": {"label": label, "reason": reason, "worst_case": num(val["cases"]["Low"]["annual_net"])},
        "model": {"records": len(df), "failures": int(df["Machine failure"].sum()), "train": res["n_train"],
                  "test": res["n_test"], "causes": {causes[k]: int(df[k].sum()) for k in causes},
                  "auc": {n: num(m["auc"]) for n, m in res["models"].items()}},
        "tracking": {"source": source, "months": t["label"].tolist(),
                     "forecast": [num(x) for x in t["forecast"]], "actual": [num(x) for x in t["actual"]],
                     "delivered": num(t["cum_actual"].iloc[-1]), "promised": num(t["cum_forecast"].iloc[-1]),
                     "plan": num(published * len(t) / 12),
                     "drift_months": t.loc[t["drift_alert"], "label"].tolist(),
                     "low_adoption_months": t.loc[t["adoption_alert"], "label"].tolist(),
                     "gap": [{"cause": g["cause"], "effect": num(g["effect"])} for _, g in gap.iterrows()]},
    }
    out = os.path.join(ROOT, "site", "assets", "data.js")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("// Generated by site/export_data.py. Do not edit by hand.\nwindow.CASE = "
                + json.dumps(case, indent=1) + ";\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
