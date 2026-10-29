"""Rank an AI provider's solution library by return on investment for one client.

Three kinds of numbers go in, and every result says which kind it rests on:
  Client's figures      what each problem costs the client a year today (example figures, replace with theirs)
  Published results     how much of that cost each AI removed elsewhere, low to high (data/ai_library.csv)
  Provider price list   setup cost, yearly running cost and months to go live for one site (illustrative),
                        multiplied by the client's deployment size (number of sites, lines or user groups)

For each solution that serves the client's industry:
  Value a year      = problem cost x improvement (low, expected = midpoint, high)
  Setup, running     = price list x deployment size
  Net a year        = value a year - running cost
  Data work         = each point the client's data falls short of what the solution needs
                      adds 20% to setup and 2 months to go live
  3-year net value  = net a year x live months in 3 years / 12 - setup
  3-year cost       = setup + running cost x live months in 3 years / 12
  ROI               = 3-year net value / 3-year cost
  Payback           = months to go live + setup / (net a year / 12)

The library is ranked by 3-year net value, either expected (midpoint) or cautious (low end of the range).
The ranking is an order of priority, not a winner: each solution fixes a different problem, and a client may buy
several. Solutions are only ever compared for one client, never across industries.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INDUSTRIES = ["Semiconductors", "Automotive", "Telecom", "Banking"]   # the provider's whole library
DATA_WORK_SETUP = 0.20      # extra setup cost per point of missing data quality
DATA_WORK_MONTHS = 2        # extra months to go live per point
HORIZON = 36                # months

# Strongest evidence first; used to show how much to trust each estimate
EVIDENCE_ORDER = ["Field study", "Academic study", "Bank case study", "Analyst research", "Vendor case study",
                  "Mixed", "Company-reported", "Vendor-reported"]
CLOSE_CALL = 0.25           # top two within 25% of each other on 3-year net value: too close to call on value alone


def evidence_strength(label: str) -> str:
    """Stronger: independent studies. Medium: analyst research and named vendor case studies. Weaker: the rest."""
    i = EVIDENCE_ORDER.index(label) if label in EVIDENCE_ORDER else len(EVIDENCE_ORDER)
    return "Stronger" if i < 3 else "Medium" if i < 5 else "Weaker"


@dataclass
class Client:
    name: str
    industry: str
    blurb: str
    data_quality: int                     # 1-5: how clean, connected and available their data is
    size: int                             # deployment size: sites, lines or user groups the price list is multiplied by
    problem_costs: dict = field(default_factory=dict)   # problem -> what it costs them a year today


# Example clients are manufacturers: the case covers the library's manufacturing entries in depth
EXAMPLE_CLIENTS = {
    # The worked example in stages 2-5: downtime matches its 40 breakdowns a year x $25k (src/value.py Costs)
    "Auto-parts plant": Client(
        "Auto-parts plant", "Automotive", "One machining plant of a tier-1 parts supplier to car makers", 4, 1,
        {"Unplanned machine downtime": 1_000_000, "Scrap, rework and warranty defects": 1_500_000,
         "Shortages, expediting and excess inventory": 1_200_000}),
    "Chip fab": Client(
        "Chip fab", "Semiconductors", "A specialty 200mm wafer fab with about $1.2B in revenue", 4, 3,
        {"Unplanned tool downtime": 6_000_000, "Scrap, rework and escaped defects": 8_000_000,
         "Wafers lost to low yield": 30_000_000, "Shortages, expediting and excess inventory": 5_000_000,
         "Facility energy bill": 25_000_000}),
}


def load_library(path: str = "data/ai_library.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def for_industry(library: pd.DataFrame, industry: str) -> pd.DataFrame:
    return library[library["industry"] == industry].reset_index(drop=True)


def _net_over_horizon(net_year: float, months_live: float, setup: float) -> float:
    return net_year * max(0.0, HORIZON - months_live) / 12 - setup


def rank(library: pd.DataFrame, client: Client, cautious: bool = False) -> pd.DataFrame:
    """Every solution for the client's industry, priced for them and ranked best first."""
    lib = for_industry(library, client.industry).copy()
    lib["problem"] = lib["problem_cost"]
    lib["problem_cost"] = lib["problem"].map(client.problem_costs).fillna(0.0)
    lib["running"] = lib["running_cost"] * client.size
    lib["improvement_mid"] = (lib["improvement_low"] + lib["improvement_high"]) / 2
    for end in ("low", "mid", "high"):
        lib[f"value_{end}"] = lib["problem_cost"] * lib[f"improvement_{end}"]
        lib[f"net_{end}"] = lib[f"value_{end}"] - lib["running"]

    lib["data_gap"] = (lib["data_needed"] - client.data_quality).clip(lower=0)
    lib["setup"] = lib["setup_cost"] * client.size * (1 + DATA_WORK_SETUP * lib["data_gap"])
    lib["months"] = lib["months_to_live"] + DATA_WORK_MONTHS * lib["data_gap"]
    live = (HORIZON - lib["months"]).clip(lower=0)
    lib["cost_3y"] = lib["setup"] + lib["running"] * live / 12
    for end in ("low", "mid", "high"):
        lib[f"net3_{end}"] = lib[f"net_{end}"] * live / 12 - lib["setup"]

    basis = "low" if cautious else "mid"
    lib["net_year"], lib["net_3y"] = lib[f"net_{basis}"], lib[f"net3_{basis}"]
    lib["roi"] = lib["net_3y"] / lib["cost_3y"]
    lib["payback"] = np.where(lib["net_year"] > 0, lib["months"] + lib["setup"] / (lib["net_year"] / 12), np.inf)
    lib["evidence_rank"] = lib["evidence"].map({e: i for i, e in enumerate(EVIDENCE_ORDER)}).fillna(len(EVIDENCE_ORDER))
    out = lib.sort_values(["net_3y", "evidence_rank"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def rank_changes(library: pd.DataFrame, client: Client) -> list[str]:
    """Solutions whose position moves between the expected and the cautious ranking."""
    exp = rank(library, client)[["solution", "rank"]]
    cau = rank(library, client, cautious=True)[["solution", "rank"]]
    both = exp.merge(cau, on="solution", suffixes=("_exp", "_cau"))
    return [f"{r.solution}: {r.rank_exp} → {r.rank_cau}" for r in both.itertuples() if r.rank_exp != r.rank_cau]


def too_close(ranked: pd.DataFrame) -> bool:
    """True when the runner-up's 3-year net value is within CLOSE_CALL of the top pick's."""
    if len(ranked) < 2 or ranked.loc[0, "net_3y"] <= 0:
        return False
    return ranked.loc[1, "net_3y"] >= (1 - CLOSE_CALL) * ranked.loc[0, "net_3y"]


def break_even_improvement(row: pd.Series) -> float:
    """Share of the problem cost the AI must remove to just pay back its 3-year cost."""
    live = max(0.0, HORIZON - row["months"])
    if row["problem_cost"] <= 0 or live == 0:
        return float("inf")
    return row["cost_3y"] / (row["problem_cost"] * live / 12)
