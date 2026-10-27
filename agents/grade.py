"""Grade saved agent runs against agents/eval_cases.json and print a scorecard.

Run: uv run --no-project --python 3.12 --with-requirements requirements.txt python -m agents.grade
"""

import json

from src.agents import cost, grade_intake, grade_memo, grade_redteam


def scorecard(runs: list[dict], cases: dict) -> dict:
    by_id = {c["id"]: c for group in cases.values() for c in group}
    rows = {"intake": [], "memo": [], "redteam": []}
    for r in runs:
        case = by_id[r["id"]]
        reply = r.get("final", r["reply"])                 # v2 agents post-process the raw reply
        if r["agent"] == "intake":
            g = grade_intake(reply, case["expected"])
        elif r["agent"] == "memo":
            g = grade_memo(reply, case["facts"], case["context"])
        else:
            g = {**grade_redteam(reply, case["planted"]), "held_out": case.get("held_out", False)}
            g["clean_high"] = g["high"] if case["planted"] is None else 0     # false alarms on the clean case
            d = json.loads(reply) if r.get("final") else {}
            g["explained"], g["n_flags"] = d.get("explained", 0), len(d.get("objections", []))
        rows[r["agent"]].append({"id": r["id"], **g, "cost": cost(r["input_tokens"], r["output_tokens"])})
    flaws = [g for g in rows["redteam"] if g["caught"] is not None and not g["held_out"]]
    held = [g for g in rows["redteam"] if g["held_out"]]
    return {
        "intake_valid_json": sum(g["valid_json"] for g in rows["intake"]) / len(rows["intake"]),
        "intake_field_accuracy": sum(g["score"] for g in rows["intake"]) / len(rows["intake"]),
        "intake_invented_fields": sum(len(g["invented"]) for g in rows["intake"]),
        "memo_faithful": sum(g["faithful"] for g in rows["memo"]),
        "memo_invented_numbers": [n for g in rows["memo"] for n in g["invented"]],
        "redteam_caught": sum(g["caught"] for g in flaws),
        "redteam_caught_first": sum(g["caught_first"] for g in flaws),
        "redteam_flaws": len(flaws),
        "redteam_held_out_caught": sum(g["caught"] for g in held),
        "redteam_held_out": len(held),
        "redteam_clean_high": sum(g["clean_high"] for g in rows["redteam"] if g["caught"] is None),
        "redteam_explained": sum(g["explained"] for g in rows["redteam"]),
        "redteam_flags": sum(g["n_flags"] for g in rows["redteam"]),
        "total_cost": sum(g["cost"] for group in rows.values() for g in group),
        "rows": rows,
    }


if __name__ == "__main__":
    cases = json.load(open("agents/eval_cases.json"))
    results = json.load(open("agents/results.json"))
    for version, runs in results.items():
        s = scorecard(runs, cases)
        print(version, {k: v for k, v in s.items() if k != "rows"})
        for g in s["rows"]["redteam"]:
            print("  ", g["id"], g["flagged"], g["caught"], g["caught_first"])
