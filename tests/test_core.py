import numpy as np
import pandas as pd

from src.readiness import verdict
from src.scoring import score
from src.tracking import explain_gap, simulate, track, validate
from src.value import Costs, bootstrap, default_ranges, evaluate, scenarios, sensitivity


def test_scoring_matches_hand_calculation():
    df = pd.read_csv("data/use_cases.csv")
    pm = score(df).set_index("use_case").loc["Predictive maintenance"]
    assert (pm["value"], pm["feasibility"], pm["priority"]) == (4.75, 3.8, 18.05)


def test_verdict_rules():
    dims = ["Data", "Workflow fit", "ROI clarity", "User trust", "Governance & risk"]
    assert verdict(dict.fromkeys(dims, 4.0))[0] == "Go"
    assert verdict({**dict.fromkeys(dims, 4.5), "Data": 2.0})[0] == "Fix first"
    assert verdict(dict.fromkeys(dims, 2.0))[0] == "Not ready"


def test_value_math():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])           # 1 caught, 1 missed, 1 false alarm
    c = Costs(failures_per_year=2, cost_missed=100, cost_caught=10, cost_false_alarm=5,
              running_cost=0, deployment_cost=1, months_to_deploy=0)
    r = evaluate(y, p, 0.5, c)
    # without AI: 2*100 = 200; with AI: 1*10 + 1*100 + 1*5 = 115
    assert round(r["annual_net"], 6) == 85


def _toy_model(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.15, n), 0, 1)   # failures score higher, with overlap
    return y, p


def test_bootstrap_range_contains_point_estimate():
    y, p = _toy_model()
    c = Costs()
    r = evaluate(y, p, 0.4, c)
    band = bootstrap(y, p, 0.4, c, n=300)
    for k in ["annual_net", "recall", "caught", "false_alarms"]:
        lo, hi = band[k]
        assert lo <= r[k] <= hi
    # payback range is ordered and lines up with the savings range
    assert band["payback_months"][0] <= r["payback_months"] <= band["payback_months"][1]


def test_sensitivity_and_scenarios_are_ordered():
    y, p = _toy_model()
    c = Costs()
    ranges = default_ranges(c)
    t = sensitivity(y, p, 0.4, c, ranges)
    assert list(t["swing"]) == sorted(t["swing"], reverse=True)
    # A pricier missed failure makes the AI worth more; a pricier deployment worth less
    row = t.set_index("key")
    assert row.loc["cost_missed", "at_high"] > row.loc["cost_missed", "at_low"]
    assert row.loc["deployment_cost", "at_high"] < row.loc["deployment_cost", "at_low"]
    cases = scenarios(y, p, 0.4, c, ranges)
    assert cases["Low"]["three_year"] <= cases["Base"]["three_year"] <= cases["High"]["three_year"]
    # Low case is at least as bad as the worst single assumption
    assert cases["Low"]["three_year"] <= t[["at_low", "at_high"]].min().min()


def _tracking_inputs():
    c = Costs(failures_per_year=12, cost_missed=100, cost_caught=10, cost_false_alarm=5,
              running_cost=120, deployment_cost=0, months_to_deploy=0)
    r = {"recall": 0.9, "false_alarms": 12, "annual_net": 12 * 0.9 * 90 - 12 * 5 - 120}
    return c, r


def test_tracking_actual_savings_and_alerts():
    c, r = _tracking_inputs()
    actuals = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "alerts_raised": [4, 4, 4, 4],
        "failures_flagged": [2, 2, 1, 1],
        "failures_prevented": [1, 2, 1, 1],
        "failures_missed": [1, 0, 1, 1],
        "false_alarms": [1, 2, 2, 2],
    })
    t = track(actuals, r, c)
    # month 1: 1 prevented x (100 - 10) - 1 false alarm x 5 - 120 / 12 = 75
    assert t.loc[0, "actual"] == 75
    assert list(t["adoption"]) == [0.5, 1.0, 0.75, 0.75]
    assert list(t["adoption_alert"]) == [True, False, False, False]          # US-6.2: below 70%
    # rolling 3-month recall: months 1-3 = 5/6 (not 10 pts below 0.9), months 2-4 = 4/6
    assert list(t["drift_alert"]) == [False, False, False, True]             # US-6.3


def test_gap_effects_add_up_to_total_gap():
    c, r = _tracking_inputs()
    for seed in range(5):
        t = track(simulate(r, c, seed=seed), r, c)
        gap = explain_gap(t, r, c)
        total = t["cum_actual"].iloc[-1] - t["cum_forecast"].iloc[-1]
        assert abs(gap["effect"].sum() - total) < 1e-6


def test_actuals_matching_forecast_have_no_gap():
    c, r = _tracking_inputs()
    # 1 failure a month, 0.9 flagged, all acted on, 1 false alarm: exactly the forecast over 10 months
    actuals = pd.DataFrame({"month": [f"m{i}" for i in range(10)], "alerts_raised": [2] * 10,
                            "failures_flagged": [1] * 9 + [0], "failures_prevented": [1] * 9 + [0],
                            "failures_missed": [0] * 9 + [1], "false_alarms": [1] * 10})
    t = track(actuals, r, c)
    assert abs(t["cum_actual"].iloc[-1] - t["cum_forecast"].iloc[-1]) < 1e-6
    assert explain_gap(t, r, c)["effect"].abs().max() < 1e-6


def test_validate_catches_bad_files():
    good = pd.read_csv("data/sample_actuals.csv")
    assert validate(good) == []
    assert "Missing columns" in validate(good.drop(columns="false_alarms"))[0]
    bad = good.copy()
    bad.loc[0, "failures_prevented"] = bad.loc[0, "failures_flagged"] + 1
    assert any("cannot exceed failures_flagged" in e for e in validate(bad))
