import numpy as np
import pandas as pd

from src.readiness import verdict
from src.scoring import score
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
