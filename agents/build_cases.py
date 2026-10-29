"""Build agents/eval_cases.json: test cases with known answers for the three agents.

Run: uv run --no-project --python 3.12 --with-requirements requirements.txt python -m agents.build_cases
"""

import json
from dataclasses import asdict, replace

from src.genai import GenAIInputs, evaluate as genai_evaluate
from src.model import train_and_score
from src.value import Costs, evaluate, sweep

# Intake: stated numbers converted to a year / minutes; None = not stated, must stay blank.
INTAKE = [
    ("Our 4-person accounts payable team processes about 300 invoices a week. Each takes 10 minutes to key in; "
     "we think AI could cut 7 minutes. Clerks cost us $38 an hour loaded.",
     dict(tasks_per_year=15_600, minutes_saved_per_task=7, review_minutes_per_task=None, hourly_cost=38, people=4)),
    ("Support team of 12 answers 1,500 tickets a month. Drafting a reply takes 8 minutes; AI drafts would save about "
     "5 minutes, and agents would spend 1 minute checking each. Loaded cost $45/hour.",
     dict(tasks_per_year=18_000, minutes_saved_per_task=5, review_minutes_per_task=1, hourly_cost=45, people=12)),
    ("Our 6 maintenance planners write roughly 40 work orders a week, half an hour each, and an AI assistant could "
     "save 15 minutes per order. Planner cost is $55/hr.",
     dict(tasks_per_year=2_080, minutes_saved_per_task=15, review_minutes_per_task=None, hourly_cost=55, people=6)),
    ("Sales ops: 3 analysts build 20 proposals a month at about 3 hours each. AI could save 2 hours per proposal but "
     "someone would need 30 minutes to review it. Analysts are $70 an hour.",
     dict(tasks_per_year=240, minutes_saved_per_task=120, review_minutes_per_task=30, hourly_cost=70, people=3)),
    ("HR screens 5,000 resumes a year with two recruiters. Each takes 6 minutes; AI summaries might save 4 minutes.",
     dict(tasks_per_year=5_000, minutes_saved_per_task=4, review_minutes_per_task=None, hourly_cost=None, people=2)),
    ("Legal reviews 600 NDAs a year, 45 minutes each. An AI first pass saves 30 minutes, but attorneys must spend "
     "10 minutes checking it. Blended cost is $120/hour for a team of 5.",
     dict(tasks_per_year=600, minutes_saved_per_task=30, review_minutes_per_task=10, hourly_cost=120, people=5)),
    ("We want to use AI for marketing emails. Not sure about volumes yet.",
     dict(tasks_per_year=None, minutes_saved_per_task=None, review_minutes_per_task=None, hourly_cost=None, people=None)),
    ("Field service writes 250 visit reports a day across 50 technicians, 12 minutes each; AI built from voice notes "
     "saves 9 minutes per report. Technicians cost $60/hr loaded and work 250 days a year.",
     dict(tasks_per_year=62_500, minutes_saved_per_task=9, review_minutes_per_task=None, hourly_cost=60, people=50)),
]


def money(x: float) -> str:
    return f"${x:,.0f}"


def genai_facts(g: GenAIInputs) -> dict:
    r = genai_evaluate(g)
    return {
        "Work orders per year": f"{g.tasks_per_year:,.0f}",
        "Planners using the tool": f"{g.people:.0f}",
        "Minutes saved per work order before review": f"{g.minutes_saved_per_task:g}",
        "Review minutes per draft": f"{g.review_minutes_per_task:g}",
        "Loaded hourly cost": money(g.hourly_cost),
        "Setup hours": f"{g.setup_hours:g}",
        "Training hours per person": f"{g.training_hours_per_person:g}",
        "Share of drafts with an error reaching the floor": f"{g.error_rate:.1%}",
        "Share of saved hours that becomes cash": f"{g.share_redeployed:.0%}",
        "Model cost per draft": f"${g.model_cost_per_task:.5f}",
        "Value of time saved per year": money(r["gain"]),
        "Total first-year cost": money(r["total_cost"]),
        "First-year ROI": f"{r['roi']:.0%}",
        "First-year cash ROI": f"{r['cash_roi']:.0%}",
    }


BASE = GenAIInputs(tasks_per_year=2_000, minutes_saved_per_task=15, hourly_cost=55, model_cost_per_task=0.000037,
                   setup_hours=40, review_minutes_per_task=4, error_rate=0.017, share_redeployed=0.5)
CONTEXT = ("6 maintenance planners at one plant each write work orders that take about 20 minutes by hand. "
           "A 30-alert pilot found the AI's draft had the wrong cause in 17% of cases and some drafts mentioned "
           "computer parts, so every draft is reviewed. The tool must be connected to the plant's maintenance system.")
FLAWS = {
    "review_minutes_per_task": dict(review_minutes_per_task=0),
    "share_redeployed": dict(share_redeployed=1.0),
    "minutes_saved_per_task": dict(minutes_saved_per_task=45),
    "tasks_per_year": dict(tasks_per_year=200_000),
    "setup_hours": dict(setup_hours=0),
    "error_rate": dict(error_rate=0.0),
}
# Held out: added after version 2 was designed, to check it was not tuned to the six flaws above.
HELD_OUT = {
    "hourly_cost": dict(hourly_cost=400),
    "training_hours_per_person": dict(training_hours_per_person=0),
}
EXTRA = {"manual_minutes": 20, "pilot_wrong_rate": 0.17}


def main():
    res = train_and_score()
    c = Costs()
    y, p = res["y_test"], res["models"]["Gradient boosting"]["proba"]
    curve = sweep(y, p, c)
    r = evaluate(y, p, float(curve.loc[curve["annual_net"].idxmax(), "threshold"]), c)
    pdm_facts = {
        "Use case": "Predictive maintenance on machining lines",
        "Unplanned failures per year today": f"{c.failures_per_year}",
        "Share of failures caught": f"{r['recall']:.0%}",
        "Share of alerts that are real failures": f"{r['precision']:.0%}",
        "Annual net savings": money(r["annual_net"]),
        "Annual net savings, 90% range": "$611,295 to $704,047",
        "Annual net savings, low case": "$163,971",
        "Payback": f"{r['payback_months']:.1f} months",
        "Deployment cost": money(c.setup),
        "Biggest driver": "Cost of an unplanned failure ($25,000)",
        "Readiness verdict": "Fix first: workflow fit and user trust score below 2.5 of 5",
    }
    cases = {
        "intake": [{"id": f"IN-{i}", "description": d, "expected": e} for i, (d, e) in enumerate(INTAKE, 1)],
        "memo": [
            {"id": "MEMO-PDM", "title": "Predictive maintenance", "context": "Manufacturing plant, model tested on 3,000 unseen records.", "facts": pdm_facts},
            {"id": "MEMO-GENAI", "title": "AI work-order assistant", "context": CONTEXT, "facts": genai_facts(BASE)},
        ],
        "redteam": [{"id": "RT-CLEAN", "title": "AI work-order assistant", "context": CONTEXT, "planted": None,
                     "facts": genai_facts(BASE), **EXTRA}]
                   + [{"id": f"RT-{k}", "title": "AI work-order assistant", "context": CONTEXT, "planted": k,
                       "facts": genai_facts(replace(BASE, **v)), **EXTRA} for k, v in FLAWS.items()]
                   + [{"id": f"RT-{k}", "title": "AI work-order assistant", "context": CONTEXT, "planted": k,
                       "held_out": True, "facts": genai_facts(replace(BASE, **v)), **EXTRA} for k, v in HELD_OUT.items()],
    }
    with open("agents/eval_cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    print({k: len(v) for k, v in cases.items()})


if __name__ == "__main__":
    main()
