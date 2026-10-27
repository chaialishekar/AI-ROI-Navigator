"""GenAI pilot: draft maintenance work orders from predictive-maintenance alerts.

The prediction model raises an alert; a small language model (Amazon Nova Micro on Bedrock)
turns it into a work order for the technician. This module picks the pilot cases, builds the
prompt and grades the drafts. The Bedrock calls themselves are in pilot/run_pilot.py.

Grading is objective: the model must name one failure mode from a fixed list, and we compare it
with the true failure mode in the AI4I 2020 data (hidden from the model).
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import RANDOM_STATE, TARGET, build_features, load_data

MODEL_ID = "us.amazon.nova-micro-v1:0"       # US inference profile: requests stay in US Regions
PRICE_PER_1K = {"input": 0.000035, "output": 0.00014}   # USD, us-east-2 on-demand (AWS Price List)
MAX_TOKENS = 500
ALERT_THRESHOLD = 0.05                       # dollar-optimal threshold from the Prove value tab

MODES = {
    "TWF": "tool wear",
    "HDF": "heat dissipation",
    "PWF": "power",
    "OSF": "overstrain",
    "NONE": "no fault found",
}
OSF_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}

SYSTEM_PROMPT = f"""You are a maintenance planner at a machining plant. A failure-prediction model has raised an alert.
Write a work order for the technician. Use only the readings given; do not invent measurements.

Maintenance manual (failure limits):
- tool wear: tools fail between 200 and 240 minutes of wear. Replace the tool.
- heat dissipation: fails when (process temp - air temp) is below 8.6 K AND speed is below 1380 rpm. Check cooling and fans.
- power: fails when power is below 3500 W or above 9000 W. Check drive and load settings.
- overstrain: fails when tool wear x torque exceeds {OSF_LIMIT['L']:,} min*Nm (L variant), {OSF_LIMIT['M']:,} (M) or {OSF_LIMIT['H']:,} (H). Reduce load or replace tool.
- If no limit is breached, the alert may be a false alarm: choose "no fault found" and order a short inspection.

Reply with JSON only, no other text:
{{"title": str, "priority": "P1" | "P2" | "P3", "failure_mode": one of {json.dumps(list(MODES.values()))},
 "reason": str (one sentence citing the readings), "steps": [str] (3-6 steps), "parts": [str],
 "safety": [str], "estimated_minutes": int}}"""


# Version 2: code decides the failure mode (diagnose), the model only writes the work order.
SYSTEM_PROMPT_V2 = f"""You are a maintenance planner at a machining plant. A failure-prediction model has raised an alert
and a rule check has already diagnosed the cause. Write a clear work order for the technician.
Keep the diagnosis exactly as given; do not change it or invent measurements.
For "no fault found", order a short inspection and say the alert may be a false alarm.

Reply with JSON only, no other text:
{{"title": str, "priority": "P1" | "P2" | "P3", "failure_mode": one of {json.dumps(list(MODES.values()))},
 "reason": str (one sentence citing the readings), "steps": [str] (3-6 steps), "parts": [str],
 "safety": [str], "estimated_minutes": int}}"""


def diagnose(case: pd.Series) -> tuple[str, str]:
    """Apply the maintenance manual's limits in code. Returns (failure mode, reason).

    Checked from most to least severe. The temperature difference is compared at one decimal, as planners read it.
    """
    diff = round(case["process_temp_k"] - case["air_temp_k"], 1)
    limit = OSF_LIMIT[case["variant"]]
    if case["power_w"] < 3500 or case["power_w"] > 9000:
        return "power", f"power {case['power_w']:.0f} W is outside the 3500-9000 W range"
    if case["wear_x_torque"] > limit:
        return "overstrain", f"wear x torque {case['wear_x_torque']:.0f} min*Nm exceeds the {limit:,} limit for variant {case['variant']}"
    if diff <= 8.6 and case["rpm"] < 1380:
        return "heat dissipation", f"temperature difference {diff:.1f} K is at or below 8.6 K at {case['rpm']:.0f} rpm"
    if case["tool_wear_min"] >= 200:
        return "tool wear", f"tool wear {case['tool_wear_min']:.0f} min is in the 200-240 min failure window"
    return "no fault found", "no limit in the maintenance manual is breached"


def rule_alerts() -> np.ndarray:
    """A no-AI alternative: alert on every test-set record that breaks a limit in the maintenance manual.

    Returns 1.0 / 0.0 per record, in the same order as train_and_score()'s test set, so it can be priced
    with value.evaluate() at any threshold between 0 and 1. In AI4I 2020 the failures were generated from
    these same limits, so this baseline is unrealistically strong on this dataset.
    """
    df = load_data()
    _, idx_te = train_test_split(df.index, test_size=0.3, stratify=df[TARGET], random_state=RANDOM_STATE)
    test = df.loc[idx_te]
    f = build_features(test)
    f["variant"] = test["Type"].values
    return np.array([diagnose(r)[0] != "no fault found" for _, r in f.iterrows()], dtype=float)


def _true_mode(row: pd.Series) -> str:
    hits = [k for k in ["TWF", "HDF", "PWF", "OSF"] if row[k] == 1]
    return "NONE" if row[TARGET] == 0 else (hits[0] if len(hits) == 1 else "MULTI")


def pilot_cases(proba: np.ndarray, n: int = 30, per_mode: int = 6, seed: int = 42) -> pd.DataFrame:
    """n alerts from the test set: up to per_mode of each failure type and false alarms, topped up at random.

    Tool-wear failures are rarely alerted on (they happen at a random wear time), so the top-up keeps
    the pilot at n cases without adding failures the model never flagged.
    proba must be the gradient-boosting test-set probabilities from train_and_score() (same split).
    """
    df = load_data()
    idx_tr, idx_te = train_test_split(df.index, test_size=0.3, stratify=df[TARGET], random_state=RANDOM_STATE)
    test = df.loc[idx_te].copy()
    test["alert_probability"] = proba
    test["true_mode"] = test.apply(_true_mode, axis=1)
    alerts = test[test["alert_probability"] >= ALERT_THRESHOLD]
    alerts = alerts[alerts["true_mode"] != "MULTI"]
    picks = pd.concat([alerts[alerts["true_mode"] == m].sample(min(per_mode, (alerts["true_mode"] == m).sum()),
                                                               random_state=seed)
                       for m in ["TWF", "HDF", "PWF", "OSF", "NONE"]])
    top_up = alerts.drop(picks.index).sample(n - len(picks), random_state=seed)
    cases = pd.concat([picks, top_up]).sample(frac=1, random_state=seed)   # shuffle so modes are mixed
    f = build_features(cases)
    return pd.DataFrame({
        "case_id": [f"WO-{i:03d}" for i in range(1, len(cases) + 1)],
        "machine": cases["Product ID"].values,
        "variant": cases["Type"].values,
        "air_temp_k": f["air_temp_k"].values,
        "process_temp_k": f["process_temp_k"].values,
        "rpm": f["rpm"].values,
        "torque_nm": f["torque_nm"].values,
        "tool_wear_min": f["tool_wear_min"].values,
        "temp_diff_k": f["temp_diff_k"].round(1).values,
        "power_w": f["power_w"].round(0).values,
        "wear_x_torque": f["wear_x_torque"].round(0).values,
        "alert_probability": cases["alert_probability"].round(3).values,
        "true_mode": cases["true_mode"].values,            # hidden from the model; used for grading
    })


def user_prompt(case: pd.Series, version: int = 1) -> str:
    text = (
        f"Alert {case['case_id']}: machine {case['machine']} (variant {case['variant']}), "
        f"failure probability {case['alert_probability']:.0%}.\n"
        f"Readings: air temp {case['air_temp_k']:.1f} K, process temp {case['process_temp_k']:.1f} K "
        f"(difference {case['temp_diff_k']:.1f} K), speed {case['rpm']:.0f} rpm, torque {case['torque_nm']:.1f} Nm, "
        f"power {case['power_w']:.0f} W, tool wear {case['tool_wear_min']:.0f} min, "
        f"wear x torque {case['wear_x_torque']:.0f} min*Nm."
    )
    if version == 2:
        mode, reason = diagnose(case)
        text += f"\nDiagnosis (rule check): {mode}, because {reason}."
    return text


def parse(text: str) -> dict | None:
    """Pull the JSON object out of the model's reply; None if it isn't valid."""
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return None


def grade(results: pd.DataFrame) -> pd.DataFrame:
    """Add pass/fail columns to pilot results (one row per call, with 'reply' and 'true_mode')."""
    out = results.copy()
    drafts = out["reply"].map(parse)
    out["valid_json"] = drafts.notna()
    out["predicted_mode"] = [d.get("failure_mode") if isinstance(d, dict) else None for d in drafts]
    out["correct_mode"] = out["predicted_mode"] == out["true_mode"].map(MODES)
    if "given_mode" in out:                      # version 2: did the model keep the rule-check diagnosis?
        out["kept_diagnosis"] = out["predicted_mode"] == out["given_mode"]
    out["cost_usd"] = (out["input_tokens"] / 1000 * PRICE_PER_1K["input"]
                       + out["output_tokens"] / 1000 * PRICE_PER_1K["output"])
    return out


def summary(graded: pd.DataFrame) -> dict:
    return {
        "calls": len(graded),
        "valid_json_rate": graded["valid_json"].mean(),
        "correct_mode_rate": graded["correct_mode"].mean(),
        "correct_on_real_failures": graded.loc[graded["true_mode"] != "NONE", "correct_mode"].mean(),
        "false_alarm_handled_rate": graded.loc[graded["true_mode"] == "NONE", "correct_mode"].mean(),
        "cost_per_task": graded["cost_usd"].mean(),
        "total_cost": graded["cost_usd"].sum(),
        "median_latency_s": graded["latency_ms"].median() / 1000,
        "avg_input_tokens": graded["input_tokens"].mean(),
        "avg_output_tokens": graded["output_tokens"].mean(),
    }
