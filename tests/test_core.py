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


def test_genai_matches_forbes_bookkeeping_example():
    # Forbes worked example: $1,440/yr tool, 10 h setup at $75, 8 h/week saved and
    # 2 h/week review at $40. Total cost $6,350, gain $16,640, ROI about 162%.
    from src.genai import GenAIInputs, evaluate as genai_evaluate
    g = GenAIInputs(tasks_per_year=52, minutes_saved_per_task=8 * 60, hourly_cost=40,
                    fixed_fees_per_year=1_440, model_cost_per_task=0, setup_hours=10, setup_rate=75,
                    review_minutes_per_task=2 * 60, training_hours_per_person=0, people=0,
                    error_rate=0, share_redeployed=1.0)
    r = genai_evaluate(g)
    assert round(r["total_cost"]) == 6_350
    assert round(r["gain"]) == 16_640
    assert round(r["net"]) == 10_290
    assert round(r["roi"] * 100) == 162
    # at break-even review time, net value is zero
    g.review_minutes_per_task = r["break_even_review_minutes"]
    assert abs(genai_evaluate(g)["net"]) < 1e-6


def test_genai_cash_vs_capacity():
    from src.genai import GenAIInputs, evaluate as genai_evaluate
    r = genai_evaluate(GenAIInputs(share_redeployed=0.25))
    assert abs(r["cash_gain"] + r["capacity_gain"] - r["gain"]) < 1e-9
    assert r["cash_roi"] < r["roi"]
    assert genai_evaluate(GenAIInputs(), year_one=False)["total_cost"] < genai_evaluate(GenAIInputs())["total_cost"]


def test_workorder_parse_and_grade():
    from src.workorder import grade, parse
    assert parse('Sure! {"failure_mode": "power"} done') == {"failure_mode": "power"}
    assert parse("no json here") is None
    results = pd.DataFrame({
        "true_mode": ["PWF", "NONE", "HDF"],
        "reply": ['{"failure_mode": "power"}', '{"failure_mode": "tool wear"}', "not json"],
        "input_tokens": [1000, 1000, 1000], "output_tokens": [1000, 1000, 1000], "latency_ms": [500, 600, 700],
    })
    g = grade(results)
    assert list(g["correct_mode"]) == [True, False, False]
    assert list(g["valid_json"]) == [True, True, False]
    assert abs(g["cost_usd"].iloc[0] - (0.000035 + 0.00014)) < 1e-12


def test_rule_diagnosis_matches_manual_and_pilot_data():
    from src.workorder import MODES, diagnose
    base = {"variant": "L", "air_temp_k": 300.0, "process_temp_k": 310.0, "rpm": 1500, "power_w": 6000,
            "wear_x_torque": 5000, "tool_wear_min": 100}
    assert diagnose(pd.Series(base))[0] == "no fault found"
    assert diagnose(pd.Series({**base, "power_w": 9100}))[0] == "power"
    assert diagnose(pd.Series({**base, "wear_x_torque": 11500}))[0] == "overstrain"
    assert diagnose(pd.Series({**base, "variant": "H", "wear_x_torque": 11500}))[0] == "no fault found"
    assert diagnose(pd.Series({**base, "process_temp_k": 308.6, "rpm": 1300}))[0] == "heat dissipation"
    assert diagnose(pd.Series({**base, "tool_wear_min": 210}))[0] == "tool wear"
    # every real failure in the pilot set is diagnosed correctly
    cases = pd.read_csv("pilot/pilot_cases.csv")
    real = cases[cases["true_mode"] != "NONE"]
    assert all(diagnose(r)[0] == MODES[r["true_mode"]] for _, r in real.iterrows())


def test_agent_graders():
    from src.agents import grade_intake, grade_memo, grade_redteam, numbers_in
    exp = {"tasks_per_year": 2080, "minutes_saved_per_task": 15, "review_minutes_per_task": None,
           "hourly_cost": 55, "people": 6}
    perfect = '{"tasks_per_year": 2080, "minutes_saved_per_task": 15, "review_minutes_per_task": null, "hourly_cost": 55, "people": 6}'
    assert grade_intake(perfect, exp)["score"] == 1
    guessed = grade_intake(perfect.replace('"review_minutes_per_task": null', '"review_minutes_per_task": 5'), exp)
    assert guessed["invented"] == ["review_minutes_per_task"] and guessed["score"] == 0.8

    assert numbers_in("Saves $660k (86%) in 2026, 3 steps, 4.7 months") == [660_000, 86]
    facts = {"Annual net savings": "$660,392", "Payback": "4.7 months"}
    assert grade_memo("We save $660k a year.", facts)["faithful"]
    assert grade_memo("We save $660k a year and $1.2M over three years.", facts)["invented"] == [1_200_000]

    reply = '{"objections": [{"assumption": "share_redeployed", "severity": "high"}, {"assumption": "setup_hours", "severity": "low"}]}'
    assert grade_redteam(reply, "share_redeployed")["caught_first"]
    assert grade_redteam(reply, "setup_hours")["caught"] is False          # low severity does not count


def test_intake_finalize_converts_units_and_drops_ungrounded_numbers():
    import json
    from src.agents import finalize_intake
    text = "Our 6 planners write 40 work orders a week; AI could save 2 hours per order. Cost is $55/hr."
    reply = json.dumps({"use_case": "work orders",
                        "volume": {"value": 40, "per": "week", "quote": "40 work orders a week"},
                        "minutes_saved": {"value": 2, "unit": "hours", "quote": "save 2 hours per order"},
                        "review_time": {"value": 5, "unit": "minutes", "quote": "5 minutes checking"},   # not in text
                        "hourly_cost": {"value": 55, "quote": "$55/hr"},
                        "people": {"value": 6, "quote": "6 planners"}})
    out = json.loads(finalize_intake(reply, text))
    assert out["tasks_per_year"] == 2080 and out["minutes_saved_per_task"] == 120
    assert out["review_minutes_per_task"] is None                       # quote not found -> left blank
    assert out["hourly_cost"] == 55 and out["people"] == 6


def test_memo_finalize_fills_placeholders_from_facts():
    from src.agents import finalize_memo
    facts = {"First-year ROI": "71%"}
    assert finalize_memo("ROI is {First-year ROI}.", facts) == "ROI is 71%."
    assert "[unknown: Payback]" in finalize_memo("Payback in {Payback}.", facts)


def test_redteam_checklist_and_code_owned_verdict():
    import json
    from src.agents import finalize_redteam_v3, plausibility_checks
    cases = {c["id"]: c for c in json.load(open("agents/eval_cases.json"))["redteam"]}
    assert plausibility_checks(cases["RT-CLEAN"]) == []                 # no false alarms on a sound case
    for cid, case in cases.items():
        if case["planted"]:
            assert plausibility_checks(case)[0]["assumption"] == case["planted"], cid
    reply = '{"objections": [{"assumption": "share_redeployed", "why": "w", "test": "t"}, {"assumption": "people", "why": "x", "test": "y"}]}'
    out = json.loads(finalize_redteam_v3(reply, cases["RT-share_redeployed"]))
    assert out["verdict"] == "send back" and out["explained"] == 1 and out["dropped"] == 1
    assert json.loads(finalize_redteam_v3('{"objections": []}', cases["RT-CLEAN"]))["verdict"] == "approve"


def test_pipeline_gold_matches_app_formula():
    """The Databricks gold table (SQL) must give the same dollars and flags as src.tracking (pandas)."""
    import json
    from src.tracking import COLUMNS, track
    gold = pd.read_csv("data/pipeline/tracked_value.csv")
    log = json.load(open("data/pipeline/run_log.json"))
    P = log["parameters"]
    c = Costs(cost_missed=P["cost_missed"], cost_caught=P["cost_caught"], cost_false_alarm=P["cost_false_alarm"],
              running_cost=P["running_cost"])
    t = track(gold[COLUMNS], {"annual_net": P["annual_net"], "recall": P["launch_recall"]}, c)
    for col in ["actual", "cum_actual", "forecast", "adoption", "recall", "recall_3m"]:
        assert np.allclose(gold[col].astype(float), t[col].astype(float), equal_nan=True), col
    assert (gold["adoption_alert"] == t["adoption_alert"]).all() and (gold["drift_alert"] == t["drift_alert"]).all()
    assert [q["month"] for q in log["quarantined"]] == ["2027-01"]          # the planted bad file


def test_matching_recommends_by_client():
    from src.matching import EXAMPLE_CLIENTS, Client, match
    lib = pd.read_csv("data/use_cases.csv")
    tops = {name: match(lib, c).iloc[0]["use_case"] for name, c in EXAMPLE_CLIENTS.items()}
    assert tops == {"Mid-size factory": "Predictive maintenance",
                    "Regional hospital network": "Clinical note summarization",
                    "Online retailer": "Customer service agent"}
    # only solutions that serve the client's industry, and every one gets reasons
    m = match(lib, EXAMPLE_CLIENTS["Online retailer"])
    assert m["serves"].str.contains("Retail").all() and m["why"].map(len).min() >= 2
    # same industry, different problem -> different recommendation
    quality = Client("Factory B", "Manufacturing", "Quality defects", 4, "Medium", "Within a year")
    assert match(lib, quality).iloc[0]["use_case"] == "Visual quality inspection"


def test_matching_data_maturity_lowers_feasibility():
    from src.matching import Client, match
    lib = pd.read_csv("data/use_cases.csv")
    rich = Client("A", "Manufacturing", "Unplanned downtime", 5, "Medium", "Within a year")
    poor = Client("B", "Manufacturing", "Unplanned downtime", 1, "Medium", "Within a year")
    f = lambda c: match(lib, c).set_index("use_case").loc["Predictive maintenance", "feasibility"]
    assert f(poor) < f(rich)
