"""Match an AI provider's solution library to one client.

The library (data/use_cases.csv) scores each solution 1-5 on value and feasibility in general.
A client profile changes that picture:

  Industry       only solutions that serve the client's industry are considered
  Main problem   a solution that solves it gets full strategic fit and +1 financial impact;
                 one that doesn't loses 2 points of strategic fit and 1 of financial impact
  Data maturity  shifts data readiness: a client at 5/5 makes data work easier, one at 1/5 harder
  Risk appetite  sets how much "low risk if the AI is wrong" weighs in feasibility
  Urgency        sets how much "speed to value" weighs in feasibility

Then the usual value x feasibility scoring ranks what is left (src/scoring.py).
"""

from dataclasses import dataclass

import pandas as pd

from src.scoring import DEFAULT_FEAS_WEIGHTS, DEFAULT_VALUE_WEIGHTS, score

INDUSTRIES = ["Manufacturing", "Retail", "Healthcare"]
RISK_WEIGHT = {"Low": 0.45, "Medium": 0.25, "High": 0.1}
URGENCY_WEIGHT = {"Within a year": 0.2, "Within 3 months": 0.4}


@dataclass
class Client:
    name: str
    industry: str
    problem: str
    data_maturity: int        # 1-5: how clean, connected and available their data is
    risk_appetite: str        # Low / Medium / High
    urgency: str              # Within a year / Within 3 months
    blurb: str = ""


EXAMPLE_CLIENTS = {
    "Mid-size factory": Client("Mid-size factory", "Manufacturing", "Unplanned downtime", 4, "Medium", "Within a year",
                               "Parts maker, 40 surprise breakdowns a year, sensor data on every machine"),
    "Regional hospital network": Client("Regional hospital network", "Healthcare", "Staff time on paperwork", 3, "Low",
                                        "Within a year", "6 hospitals, clinicians spend hours a day on documentation"),
    "Online retailer": Client("Online retailer", "Retail", "Customer service backlog", 5, "High", "Within 3 months",
                              "Growing fast, support queue doubles in peak season"),
}


def problems(library: pd.DataFrame, industry: str | None = None) -> list[str]:
    lib = library if industry is None else library[library["serves"].str.contains(industry)]
    return sorted(lib["solves"].unique())


def weights(client: Client) -> tuple[dict, dict]:
    feas = dict(DEFAULT_FEAS_WEIGHTS)
    feas["low_risk"] = RISK_WEIGHT[client.risk_appetite]
    feas["speed_to_value"] = URGENCY_WEIGHT[client.urgency]
    return dict(DEFAULT_VALUE_WEIGHTS), feas


def reasons(row: pd.Series, client: Client) -> list[str]:
    out = []
    if row["solves"] == client.problem:
        out.append(f"Solves their main problem: {client.problem.lower()}")
    else:
        out.append(f"Targets {row['solves'].lower()}, not their main problem ({client.problem.lower()})")
    needed = 6 - row["data_readiness_library"]
    out.append("Their data is good enough to start" if client.data_maturity >= needed
               else f"Needs better data than they have today ({client.data_maturity}/5, needs about {needed}/5)")
    if row["low_risk"] <= 2 and client.risk_appetite == "Low":
        out.append("A wrong answer could do real harm, and this client is cautious")
    elif row["low_risk"] >= 4:
        out.append("Little harm if the AI is wrong")
    if client.urgency == "Within 3 months":
        out.append("Shows results fast" if row["speed_to_value"] >= 4 else "Slow to show results for an urgent client")
    return out


def match(library: pd.DataFrame, client: Client) -> pd.DataFrame:
    """Solutions that serve this client, rescored for them and ranked best fit first, with reasons."""
    lib = library[library["serves"].str.contains(client.industry)].copy()
    lib["data_readiness_library"] = lib["data_readiness"]
    hit = lib["solves"] == client.problem
    lib.loc[hit, "strategic_fit"] = 5
    lib.loc[hit, "financial_impact"] = (lib.loc[hit, "financial_impact"] + 1).clip(upper=5)
    lib.loc[~hit, "strategic_fit"] = (lib.loc[~hit, "strategic_fit"] - 2).clip(lower=1)
    lib.loc[~hit, "financial_impact"] = (lib.loc[~hit, "financial_impact"] - 1).clip(lower=1)
    lib["data_readiness"] = (lib["data_readiness"] + client.data_maturity - 3).clip(1, 5)
    value_w, feas_w = weights(client)
    out = score(lib, value_w, feas_w)
    out["solves_problem"] = out["solves"] == client.problem
    out["why"] = [reasons(r, client) for _, r in out.iterrows()]
    return out
