"""Pilot-to-production readiness: 10 questions across 5 dimensions, rated 1-5."""

DIMENSIONS = {
    "Data": [
        "Sensor and maintenance data is collected consistently for the machines in scope.",
        "Past failures are recorded with dates and causes (labels the model can learn from).",
    ],
    "Workflow fit": [
        "Alerts can flow into the tools technicians already use (e.g. the maintenance system).",
        "There is a clear owner and process for acting on an alert.",
    ],
    "ROI clarity": [
        "We know what an unplanned failure costs us (downtime, repair, scrap).",
        "Finance agrees on how savings will be measured.",
    ],
    "User trust": [
        "Technicians were involved in the pilot and understand why alerts fire.",
        "Leaders will back acting on alerts even when a machine looks fine.",
    ],
    "Governance & risk": [
        "Someone is accountable for model performance after launch.",
        "There is a plan to monitor accuracy and retrain when it drifts.",
    ],
}

ACTIONS = {
    "Data": "Run a 4–6 week data-readiness sprint: standardize sensor logging and back-fill failure records.",
    "Workflow fit": "Map the alert-to-repair workflow and integrate alerts into the maintenance system before launch.",
    "ROI clarity": "Agree the cost of downtime and the savings formula with finance, and set a baseline period.",
    "User trust": "Co-design alerts with technicians, show the top reasons behind each alert, and start with one line.",
    "Governance & risk": "Name a model owner, set an accuracy floor that triggers review, and schedule retraining.",
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
