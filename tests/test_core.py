import numpy as np
import pandas as pd

from src.readiness import verdict
from src.scoring import score
from src.value import Costs, evaluate


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
