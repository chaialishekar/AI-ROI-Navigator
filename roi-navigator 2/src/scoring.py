"""Use-case prioritization: score each AI use case on Value and Feasibility.

Every criterion is scored 1-5, where 5 is always "better":
  Value       = financial_impact, strategic_fit, scale
  Feasibility = data_readiness, ease_of_build, low_risk, speed_to_value

Weights are adjustable in the app so a user can reflect their own priorities.
"""

import pandas as pd

VALUE_CRITERIA = {
    "financial_impact": "Financial impact",
    "strategic_fit": "Strategic fit",
    "scale": "Scale (users / volume)",
}
FEASIBILITY_CRITERIA = {
    "data_readiness": "Data readiness",
    "ease_of_build": "Ease of build",
    "low_risk": "Low risk if AI is wrong",
    "speed_to_value": "Speed to value",
}

DEFAULT_VALUE_WEIGHTS = {"financial_impact": 0.5, "strategic_fit": 0.25, "scale": 0.25}
DEFAULT_FEAS_WEIGHTS = {"data_readiness": 0.35, "ease_of_build": 0.2, "low_risk": 0.25, "speed_to_value": 0.2}

# Midpoint of the 1-5 scale splits the 2x2 grid
THRESHOLD = 3.5


def _weighted(df: pd.DataFrame, weights: dict) -> pd.Series:
    total = sum(weights.values()) or 1
    return sum(df[col] * w for col, w in weights.items()) / total


def quadrant(value: float, feasibility: float, threshold: float = THRESHOLD) -> str:
    if value >= threshold and feasibility >= threshold:
        return "Quick win"
    if value >= threshold:
        return "Big bet"
    if feasibility >= threshold:
        return "Fill-in"
    return "Deprioritize"


NEXT_STEP = {
    "Quick win": "Pilot now: high value and ready to build.",
    "Big bet": "Fix the blocker first (data, risk or complexity), then build.",
    "Fill-in": "Do when there is spare capacity.",
    "Deprioritize": "Park it and revisit next planning cycle.",
}


def score(df: pd.DataFrame, value_w: dict = None, feas_w: dict = None) -> pd.DataFrame:
    value_w = value_w or DEFAULT_VALUE_WEIGHTS
    feas_w = feas_w or DEFAULT_FEAS_WEIGHTS
    out = df.copy()
    out["value"] = _weighted(out, value_w).round(2)
    out["feasibility"] = _weighted(out, feas_w).round(2)
    out["priority"] = (out["value"] * out["feasibility"]).round(2)
    out["quadrant"] = [quadrant(v, f) for v, f in zip(out["value"], out["feasibility"])]
    out["next_step"] = out["quadrant"].map(NEXT_STEP)
    return out.sort_values("priority", ascending=False).reset_index(drop=True)
