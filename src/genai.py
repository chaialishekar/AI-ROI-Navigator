"""ROI of a generative AI assistant: time saved minus every cost of using it.

Gains: minutes saved per task × tasks per year × loaded hourly cost.
Costs follow the five categories in Dogra, "How To Measure AI ROI In A Small Business" (Forbes, 2026):
  1. Subscriptions and usage   fixed fees + model cost per task (from the Bedrock pilot) × tasks
  2. Setup                     hours connecting the tool and writing its instructions (one-time)
  3. Review                    minutes people spend checking each AI draft
  4. Training                  hours per person who uses the tool daily (one-time)
  5. Errors                    share of drafts with a mistake that reaches the floor × cost to fix

ROI = (gains − total costs) / total costs.

Saved hours only become cash if they go somewhere (other work, a delayed hire). `share_redeployed`
splits the gain into cash and capacity, and cash ROI uses the cash part only.
"""

from dataclasses import dataclass


@dataclass
class GenAIInputs:
    tasks_per_year: float = 2_000             # maintenance work orders written per year
    minutes_saved_per_task: float = 15        # writing time saved, before review
    hourly_cost: float = 55                   # loaded cost of a maintenance planner
    fixed_fees_per_year: float = 0            # seats, platform or hosting fees
    model_cost_per_task: float = 0.001        # Bedrock token cost per draft (replace with pilot result)
    setup_hours: float = 40                   # integration and prompt writing
    setup_rate: float = 75                    # hourly cost of whoever does the setup
    review_minutes_per_task: float = 4        # checking and fixing each draft
    training_hours_per_person: float = 2
    people: float = 6
    error_rate: float = 0.02                  # share of drafts with a mistake that reaches the floor
    cost_per_error: float = 150               # rework or a wrong part ordered
    share_redeployed: float = 0.5             # share of saved hours that turns into cash


COST_LABELS = {
    "usage": "Subscriptions and usage",
    "setup": "Setup",
    "review": "Review time",
    "training": "Training",
    "errors": "Errors",
}


def evaluate(g: GenAIInputs, year_one: bool = True) -> dict:
    """First-year ROI by default; year_one=False drops the one-time setup and training costs."""
    hours_saved = g.tasks_per_year * g.minutes_saved_per_task / 60
    review_hours = g.tasks_per_year * g.review_minutes_per_task / 60
    costs = {
        "usage": g.fixed_fees_per_year + g.model_cost_per_task * g.tasks_per_year,
        "setup": g.setup_hours * g.setup_rate if year_one else 0.0,
        "review": review_hours * g.hourly_cost,
        "training": g.training_hours_per_person * g.people * g.hourly_cost if year_one else 0.0,
        "errors": g.error_rate * g.tasks_per_year * g.cost_per_error,
    }
    total_cost = sum(costs.values())
    gain = hours_saved * g.hourly_cost
    cash_gain = gain * g.share_redeployed
    return {
        "costs": costs,
        "total_cost": total_cost,
        "gain": gain,
        "cash_gain": cash_gain,
        "capacity_gain": gain - cash_gain,
        "net": gain - total_cost,
        "roi": (gain - total_cost) / total_cost if total_cost else float("inf"),
        "cash_net": cash_gain - total_cost,
        "cash_roi": (cash_gain - total_cost) / total_cost if total_cost else float("inf"),
        "hours_saved": hours_saved,
        "review_hours": review_hours,
        "net_hours_saved": hours_saved - review_hours,
        # review minutes per task at which the assistant stops paying for itself
        "break_even_review_minutes": max(
            (gain - (total_cost - costs["review"])) / g.tasks_per_year / g.hourly_cost * 60, 0.0),
    }
