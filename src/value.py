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

Two kinds of uncertainty are reported:
  - Model uncertainty: the test set holds only ~100 real failures, so the catch and
    false-alarm rates are estimates. A bootstrap (resampling the test set) gives a range.
  - Assumption uncertainty: plant costs are estimates. Each one is moved between a low and
    high value to see which assumptions matter most, and to build low / base / high cases.
"""

from dataclasses import dataclass, replace

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
    payback_months = payback(annual_net, c)

    return {
        **m, "threshold": threshold, "recall": recall, "precision": precision, "accuracy": accuracy,
        "caught": caught, "missed": missed, "false_alarms": false_alarms,
        "cost_without": cost_without, "cost_with": cost_with, "annual_net": annual_net,
        "first_year": first_year, "three_year": three_year, "roi_3y": roi_3y,
        "payback_months": payback_months,
    }


def payback(annual_net: float, c: Costs) -> float:
    """Months from project start until savings cover the deployment cost."""
    return (c.months_to_deploy + c.deployment_cost / (annual_net / 12)) if annual_net > 0 else float("inf")


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


# ---------------------------------------------------------------- Model uncertainty

def bootstrap(y_true: np.ndarray, proba: np.ndarray, threshold: float, c: Costs,
              n: int = 1000, level: float = 0.9, seed: int = 42) -> dict:
    """Range of results if the test set had been a different sample of machines.

    Resamples the test records with replacement `n` times and returns the
    central `level` interval (default 90%) for each key as (low, high).
    Payback and ROI follow directly from annual net savings, so their ranges are
    derived from it.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    samples = pd.DataFrame([evaluate(y_true[i], proba[i], threshold, c) for i in idx])
    tail = (1 - level) / 2
    keys = ["annual_net", "three_year", "recall", "precision", "caught", "missed", "false_alarms"]
    out = {k: (samples[k].quantile(tail), samples[k].quantile(1 - tail)) for k in keys}
    net_lo, net_hi = out["annual_net"]
    out["payback_months"] = (payback(net_hi, c), payback(net_lo, c))   # more savings = faster payback
    out["roi_3y"] = tuple(v / c.deployment_cost for v in out["three_year"])
    return out


# ---------------------------------------------------------------- Assumption uncertainty

ASSUMPTIONS = {
    "failures_per_year": "Unplanned failures per year",
    "cost_missed": "Cost of an unplanned failure ($)",
    "cost_caught": "Cost of a planned repair ($)",
    "cost_false_alarm": "Cost of a false alarm ($)",
    "running_cost": "Yearly running cost of the AI ($)",
    "deployment_cost": "One-time deployment cost ($)",
    "months_to_deploy": "Months until live",
}

# Default spread around each base value: (share below, share above).
# Costs that are often underestimated (failures, deployment, timeline) get a wider upside.
DEFAULT_SPREAD = {
    "failures_per_year": (0.25, 0.25),
    "cost_missed": (0.40, 0.40),
    "cost_caught": (0.25, 0.25),
    "cost_false_alarm": (0.50, 0.50),
    "running_cost": (0.25, 0.50),
    "deployment_cost": (0.25, 0.50),
    "months_to_deploy": (0.50, 1.00),
}


def default_ranges(c: Costs) -> dict:
    """Low and high value for each assumption, around the base value in `c`."""
    ranges = {}
    for k, (down, up) in DEFAULT_SPREAD.items():
        base = getattr(c, k)
        ranges[k] = (base * (1 - down), base * (1 + up))
    return ranges


def sensitivity(y_true: np.ndarray, proba: np.ndarray, threshold: float, c: Costs,
                ranges: dict, metric: str = "three_year") -> pd.DataFrame:
    """Move one assumption at a time to its low and high value (tornado analysis).

    Uses 3-year net value by default because, unlike annual savings, it reflects every
    assumption including deployment cost and timeline. Sorted biggest swing first.
    """
    base = evaluate(y_true, proba, threshold, c)[metric]
    rows = []
    for k, (lo, hi) in ranges.items():
        at_lo = evaluate(y_true, proba, threshold, replace(c, **{k: lo}))[metric]
        at_hi = evaluate(y_true, proba, threshold, replace(c, **{k: hi}))[metric]
        rows.append({"key": k, "assumption": ASSUMPTIONS[k], "low_input": lo, "high_input": hi,
                     "at_low": at_lo, "at_high": at_hi, "base": base, "swing": abs(at_hi - at_lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def scenarios(y_true: np.ndarray, proba: np.ndarray, threshold: float, c: Costs, ranges: dict) -> dict:
    """Low / base / high cases.

    Low case = every assumption at whichever end of its range hurts 3-year value;
    high case = every assumption at the end that helps it.
    """
    worst, best = {}, {}
    for _, r in sensitivity(y_true, proba, threshold, c, ranges).iterrows():
        lo_hurts = r["at_low"] <= r["at_high"]
        worst[r["key"]] = r["low_input"] if lo_hurts else r["high_input"]
        best[r["key"]] = r["high_input"] if lo_hurts else r["low_input"]
    return {
        "Low": evaluate(y_true, proba, threshold, replace(c, **worst)),
        "Base": evaluate(y_true, proba, threshold, c),
        "High": evaluate(y_true, proba, threshold, replace(c, **best)),
    }
