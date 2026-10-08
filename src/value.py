"""Turn model results into dollars.

At a given alert threshold the model's test-set results give:
  caught failures (TP), missed failures (FN), false alarms (FP).
These are scaled to a plant using the number of failures it has per year without AI.

Annual cost without AI  = failures × cost of an unplanned failure
Annual cost with AI     = caught × cost of a planned repair
                        + missed × cost of an unplanned failure
                        + false alarms × cost of an inspection
                        + yearly running cost of the AI
Annual net savings      = cost without AI − cost with AI
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    failures_per_year: float = 40          # unplanned failures today, no AI
    cost_missed: float = 25_000            # unplanned breakdown: downtime + emergency repair
    cost_caught: float = 4_000             # planned repair when caught early
    cost_false_alarm: float = 500          # technician inspection that finds nothing
    running_cost: float = 60_000           # yearly licence, compute, support
    deployment_cost: float = 150_000       # one-time setup and integration
    months_to_deploy: float = 2            # until the model is live


def confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = proba >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fn = int(np.sum(~pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float, c: Costs) -> dict:
    m = confusion(y_true, proba, threshold)
    actual = m["tp"] + m["fn"]
    recall = m["tp"] / actual if actual else 0.0
    precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
    accuracy = (m["tp"] + m["tn"]) / len(y_true)

    scale = c.failures_per_year / actual if actual else 0.0
    caught = m["tp"] * scale
    missed = m["fn"] * scale
    false_alarms = m["fp"] * scale

    cost_without = c.failures_per_year * c.cost_missed
    cost_with = (caught * c.cost_caught + missed * c.cost_missed
                 + false_alarms * c.cost_false_alarm + c.running_cost)
    annual_net = cost_without - cost_with

    live_months_y1 = max(0.0, 12 - c.months_to_deploy)
    first_year = annual_net * live_months_y1 / 12 - c.deployment_cost
    three_year = annual_net * max(0.0, 36 - c.months_to_deploy) / 12 - c.deployment_cost
    roi_3y = three_year / c.deployment_cost if c.deployment_cost else float("nan")
    payback_months = (c.months_to_deploy + c.deployment_cost / (annual_net / 12)) if annual_net > 0 else float("inf")

    return {
        **m, "threshold": threshold, "recall": recall, "precision": precision, "accuracy": accuracy,
        "caught": caught, "missed": missed, "false_alarms": false_alarms,
        "cost_without": cost_without, "cost_with": cost_with, "annual_net": annual_net,
        "first_year": first_year, "three_year": three_year, "roi_3y": roi_3y,
        "payback_months": payback_months,
    }


def sweep(y_true: np.ndarray, proba: np.ndarray, c: Costs, steps: int = 99) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, steps)
    return pd.DataFrame([evaluate(y_true, proba, t, c) for t in thresholds])


def plain_language(r: dict) -> str:
    caught_of_10 = round(r["recall"] * 10)
    if r["tp"] + r["fp"] == 0:
        alarm_txt = "it raises no alerts"
    elif r["precision"] >= 0.3:
        alarm_txt = f"about {round(r['precision'] * 10)} in 10 alerts are real failures"
    else:
        alarm_txt = f"only about 1 in {round(1 / r['precision'])} alerts is a real failure"
    return f"Catches about {caught_of_10} of every 10 failures, and {alarm_txt}."
