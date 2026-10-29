"""Track value after go-live: forecast vs actual.

Each month the plant reports counts from its maintenance system:
  alerts_raised       alerts the model sent
  failures_flagged    failures the model warned about in advance (acted on or not)
  failures_prevented  flagged failures the crew fixed in time (planned repair)
  failures_missed     unplanned failures (not flagged, or flagged but ignored)
  false_alarms        inspections that found nothing wrong

From these:
  failures       = prevented + missed                        what would have broken without AI
  recall         = flagged / failures                        is the model still catching failures?
  adoption       = (prevented + false_alarms) / alerts       share of alerts the crew acted on
  actual savings = prevented × (cost missed − cost caught)
                 − false alarms × cost of a false alarm − running cost / 12
This is the same formula as the forecast in value.py, so the two compare like for like.
"""

import numpy as np
import pandas as pd

from src.value import Costs

COLUMNS = ["month", "alerts_raised", "failures_flagged", "failures_prevented", "failures_missed", "false_alarms"]
ADOPTION_FLOOR = 0.70        # US-6.2: warn when fewer than 70% of alerts are acted on
DRIFT_DROP = 0.10            # US-6.3: warn when recall is 10 points below launch
DRIFT_WINDOW = 3             # months; ~3 failures a month is too few to judge recall alone


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with an uploaded actuals file (empty = OK)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    if df.empty:
        return ["The file has no rows."]
    errors = []
    counts = df[COLUMNS[1:]]
    if not all(pd.api.types.is_numeric_dtype(counts[c]) for c in counts) or counts.isna().any().any():
        return ["Every count must be a number (no blanks)."]
    if (counts < 0).any().any():
        errors.append("Counts cannot be negative.")
    if (df["failures_prevented"] > df["failures_flagged"]).any():
        errors.append("failures_prevented cannot exceed failures_flagged.")
    if (df["failures_flagged"] > df["failures_prevented"] + df["failures_missed"]).any():
        errors.append("failures_flagged cannot exceed total failures (prevented + missed).")
    if (df["failures_prevented"] + df["false_alarms"] > df["alerts_raised"]).any():
        errors.append("Alerts acted on (prevented + false alarms) cannot exceed alerts_raised.")
    if df["month"].duplicated().any():
        errors.append("Each month should appear once.")
    return errors


def month_label(month: str, first: str) -> str:
    """'Month 3' for the third month after go-live. Simulated data is dated, but shown relative to go-live."""
    return f"Month {(pd.Period(month, 'M') - pd.Period(first, 'M')).n + 1}"


def track(actuals: pd.DataFrame, r: dict, c: Costs) -> pd.DataFrame:
    """Monthly forecast vs actual savings, adoption and drift flags.

    r is the evaluate() result the business case was approved on; its recall is the launch level.
    """
    df = actuals[COLUMNS].copy().reset_index(drop=True)
    df["label"] = [month_label(m, df["month"].iloc[0]) for m in df["month"]]
    df["failures"] = df["failures_prevented"] + df["failures_missed"]
    df["acted_on"] = df["failures_prevented"] + df["false_alarms"]
    df["recall"] = df["failures_flagged"] / df["failures"].replace(0, np.nan)
    df["adoption"] = df["acted_on"] / df["alerts_raised"].replace(0, np.nan)

    roll = df[["failures_flagged", "failures"]].rolling(DRIFT_WINDOW, min_periods=DRIFT_WINDOW).sum()
    df["recall_3m"] = roll["failures_flagged"] / roll["failures"].replace(0, np.nan)

    df["forecast"] = r["annual_net"] / 12
    df["actual"] = (df["failures_prevented"] * (c.cost_missed - c.cost_caught)
                    - df["false_alarms"] * c.cost_false_alarm - c.running_cost / 12)
    df["cum_forecast"] = df["forecast"].cumsum()
    df["cum_actual"] = df["actual"].cumsum()

    df["adoption_alert"] = df["adoption"] < ADOPTION_FLOOR
    df["drift_alert"] = df["recall_3m"] < r["recall"] - DRIFT_DROP
    return df


def _savings(failures, recall, acted, false_alarms, months, c: Costs) -> float:
    return (failures * recall * acted * (c.cost_missed - c.cost_caught)
            - false_alarms * c.cost_false_alarm - months * c.running_cost / 12)


def explain_gap(t: pd.DataFrame, r: dict, c: Costs) -> pd.DataFrame:
    """Split (actual − forecast) to date into four causes, changed one at a time.

    The forecast assumes the plant's failure rate, the model's test-set recall, every alert
    acted on and the test-set false-alarm rate. Swapping each for its actual value in turn
    gives effects that add up exactly to the total gap.
    """
    n = len(t)
    f0, rec0, acted0, fa0 = c.failures_per_year * n / 12, r["recall"], 1.0, r["false_alarms"] * n / 12
    f1 = t["failures"].sum()
    flagged = t["failures_flagged"].sum()
    rec1 = flagged / f1 if f1 else rec0
    acted1 = t["failures_prevented"].sum() / flagged if flagged else acted0
    fa1 = t["false_alarms"].sum()

    causes = ["Failures (volume)", "Model recall", "Flagged failures acted on", "False alarms"]
    assumed, actual = [f0, rec0, acted0, fa0], [f1, rec1, acted1, fa1]
    rows, current = [], list(assumed)
    for i, cause in enumerate(causes):
        before = _savings(*current, n, c)
        current[i] = actual[i]
        rows.append({"cause": cause, "assumed": assumed[i], "actual": actual[i],
                     "effect": _savings(*current, n, c) - before})
    return pd.DataFrame(rows)


def simulate(r: dict, c: Costs, n_months: int = 12, start: str = "2026-01", seed: int = 7) -> pd.DataFrame:
    """Sample rollout for demos: slow adoption at first, then the model drifts from month 9.

    Simulated data, not a real plant. Counts are random draws around the forecast rates.
    """
    rng = np.random.default_rng(seed)
    adoption = np.clip([0.55, 0.62, 0.68, 0.78, 0.85, 0.9] + [0.92] * (n_months - 6), 0, 1)[:n_months]
    recall = np.array([r["recall"]] * 8 + [r["recall"] - 0.08 * k for k in range(1, n_months - 7)])[:n_months]
    rows = []
    for m, month in enumerate(pd.period_range(start, periods=n_months, freq="M")):
        failures = rng.poisson(c.failures_per_year / 12)
        flagged = rng.binomial(failures, np.clip(recall[m], 0, 1))
        prevented = rng.binomial(flagged, adoption[m])
        false_alerts = rng.poisson(r["false_alarms"] / 12)
        rows.append({
            "month": str(month),
            "alerts_raised": flagged + false_alerts,
            "failures_flagged": flagged,
            "failures_prevented": prevented,
            "failures_missed": failures - prevented,
            "false_alarms": rng.binomial(false_alerts, adoption[m]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)
