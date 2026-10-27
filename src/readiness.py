"""Pilot-to-production readiness: 10 questions across 5 dimensions, rated 1-5."""

DIMENSIONS = {
    "Data": [
        "Sensor and maintenance data is collected consistently for the machines in scope.",
        "Past breakdowns are recorded with dates and causes.",
    ],
    "Fits how people work": [
        "Warnings show up in the tools technicians already use, such as the maintenance system.",
        "It is clear who acts on a warning, and how.",
    ],
    "Clear payoff": [
        "We know what an unplanned failure costs us (downtime, repair, scrap).",
        "Finance agrees on how savings will be measured.",
    ],
    "People trust it": [
        "Technicians helped test it and understand why warnings appear.",
        "Managers will back acting on a warning even when a machine looks fine.",
    ],
    "Clear ownership": [
        "One person owns how well the AI works after launch.",
        "There is a plan to check its accuracy and update it when it slips.",
    ],
}

ACTIONS = {
    "Data": "Spend 4–6 weeks cleaning up sensor logs and filling in past breakdown records.",
    "Fits how people work": "Map how a warning becomes a repair, and send warnings straight into the maintenance system before launch.",
    "Clear payoff": "Agree with finance what a breakdown costs and how savings will be counted.",
    "People trust it": "Design the warnings with technicians, show why each one appears, and start with one production line.",
    "Clear ownership": "Name one owner for the AI, agree the accuracy level that triggers a review, and schedule regular updates.",
}


def verdict(scores: dict) -> tuple[str, list[tuple[str, float]]]:
    """scores: {dimension: average 1-5}. Returns (verdict, 3 weakest dimensions)."""
    avg = sum(scores.values()) / len(scores)
    weakest = sorted(scores.items(), key=lambda kv: kv[1])[:3]
    if avg < 2.5:
        v = "Not ready"
    elif min(scores.values()) < 2.5:
        v = "Fix first"
    elif avg >= 3.5:
        v = "Go"
    else:
        v = "Fix first"
    return v, weakest
