"""Three analyst agents on Amazon Bedrock (Nova Lite), each with an automatic eval.

  Intake   plain-English use case  -> calculator inputs (blank when not stated, never guessed)
  Memo     business case facts     -> one-page memo for the CFO
  Red team business case facts     -> the objections a CFO would raise, tied to specific assumptions

Evals (agents/eval_cases.json, graded in code, no human judgement):
  Intake   stated numbers extracted correctly; unstated numbers left blank
  Memo     every number in the memo traces back to a fact (no invented numbers)
  Red team catches a deliberately planted flaw in otherwise reasonable business cases

Versions: v1 the model does everything; v2 code converts units, fills in memo numbers and computes ratios;
v3 (red team only) a code checklist decides what is implausible and the model explains each flag.

The app shows saved results (agents/results.json) and runs live only where AWS credentials exist.
"""

import json
import re

from src.workorder import parse

MODEL_ID = "us.amazon.nova-lite-v1:0"
PRICE_PER_1K = {"input": 0.00006, "output": 0.00024}   # USD, us-east-2 on-demand (AWS Price List)
MAX_TOKENS = {"intake": 400, "memo": 700, "redteam": 700}

INTAKE_FIELDS = ["tasks_per_year", "minutes_saved_per_task", "review_minutes_per_task", "hourly_cost", "people"]
ASSUMPTIONS = {
    "tasks_per_year": "Tasks per year",
    "minutes_saved_per_task": "Minutes saved per task",
    "review_minutes_per_task": "Review minutes per task",
    "hourly_cost": "Loaded hourly cost",
    "people": "People using the tool",
    "setup_hours": "Setup hours",
    "training_hours_per_person": "Training hours per person",
    "error_rate": "Error rate reaching the customer or floor",
    "share_redeployed": "Share of saved hours that becomes cash",
    "model_cost_per_task": "Model cost per task",
}

SYSTEM = {
    "intake": f"""You turn a plain-English description of an AI use case into inputs for an ROI calculator.
Rules:
- Convert volumes to tasks per YEAR (week x 52, month x 12, working day x the working days stated, or x 250 if they say "working days" without a number).
- Convert times to MINUTES (half an hour = 30, 2 hours = 120).
- minutes_saved_per_task is the time the AI saves, not the original task time.
- If a number is not stated, use null. Never guess. Add a short question for each null.
Reply with JSON only:
{{"use_case": str, {", ".join(f'"{f}": number | null' for f in INTAKE_FIELDS)}, "questions": [str]}}""",

    "memo": """You are an AI business analyst writing a one-page memo to the CFO.
Use ONLY the facts provided. Do not introduce any number that is not in the facts, and do not compute new ones.
Structure, in markdown, with these exact headings:
## Recommendation
## Value
## Risks
## Next step
Keep it under 250 words. Plain language, no jargon.""",

    "redteam": f"""You are a skeptical CFO reviewing an AI business case. Find the assumptions most likely to be wrong
or to overstate the value. Check every input against the context: is it realistic for this team and task?
Reply with JSON only:
{{"objections": [{{"assumption": one of {json.dumps(list(ASSUMPTIONS))},
  "severity": "high" | "medium" | "low", "issue": str (cite the number), "test": str (how to verify it in 30 days)}}],
 "verdict": "approve" | "approve with conditions" | "send back"}}
List at most 5 objections, most severe first.""",
}


# Version 2: code does the arithmetic and checks; the model extracts, writes and judges.
SYSTEM_V2 = {
    "intake": """You extract the numbers a person stated about an AI use case. Do NOT calculate or convert anything.
For each item give the number exactly as stated, its unit, and the exact words you took it from ("quote", copied
character for character from the text). If the item is not stated, use null for value and quote.
- volume: how many tasks, with "per" = year | month | week | day
- working_days: working days per year, only if stated
- minutes_saved: time the AI saves per task (not the original task time), unit = minutes | hours
- review_time: time a person spends CHECKING the AI's output per task. The original task time ("takes 10 minutes",
  "45 minutes each") is NOT review time; use null unless checking or reviewing is mentioned.
- hourly_cost: loaded cost per hour in dollars
- people: number of people doing the task
Reply with JSON only:
{"use_case": str,
 "volume": {"value": number|null, "per": str|null, "quote": str|null},
 "working_days": {"value": number|null, "quote": str|null},
 "minutes_saved": {"value": number|null, "unit": str|null, "quote": str|null},
 "review_time": {"value": number|null, "unit": str|null, "quote": str|null},
 "hourly_cost": {"value": number|null, "quote": str|null},
 "people": {"value": number|null, "quote": str|null},
 "questions": [str]}""",

    "memo": """You are an AI business analyst writing a one-page memo to the CFO.
Never type a number yourself. Every number must be a placeholder with the exact fact name in curly braces,
for example {First-year ROI} or {Value of time saved per year}. Use only fact names from the list.
If the cash ROI is negative, say so under Risks and make the recommendation conditional.
Structure, in markdown, with these exact headings:
## Recommendation
## Value
## Risks
## Next step
Keep it under 250 words. Plain language, no jargon.""",

    "redteam": f"""You are a skeptical CFO reviewing an AI business case. You get the inputs and a set of ratios an
analyst computed from them. Go through every ratio and ask: is this realistic for this team and task?
An impossible or implausible ratio (for example, saving more time than the task takes, or one person doing far more
tasks a day than possible) is HIGH severity. Then check each input against the context.
Use only these assumption names, exactly: {json.dumps(list(ASSUMPTIONS))}.
Reply with JSON only:
{{"objections": [{{"assumption": str, "severity": "high" | "medium" | "low", "issue": str (cite the number),
  "test": str (how to verify it in 30 days)}}],
 "verdict": "approve" | "approve with conditions" | "send back"}}
List at most 5 objections, most severe first.""",
}


# Version 3 changes only the red team: a code checklist decides what is implausible and how severe it is;
# the model explains each flag to the CFO and proposes a 30-day test. Intake and memo are unchanged from v2.
SYSTEM_V3 = {**SYSTEM_V2, "redteam": """You are a skeptical CFO. An analyst's checklist has already flagged the inputs listed under
"Flags" as implausible. For each flag write:
- "why": one or two plain-language sentences on how this input could overstate the value or hide a cost
- "test": one concrete way to verify the number within 30 days
Do not add flags, drop flags or change severities. Use only numbers that appear in the flag or the facts.
If there are no flags, reply {"objections": []}.
Reply with JSON only: {"objections": [{"assumption": str, "why": str, "test": str}]}"""}
PROMPTS = {1: SYSTEM, 2: SYSTEM_V2, 3: SYSTEM_V3}


def _num(text: str) -> float:
    return float(str(text).replace("$", "").replace(",", "").replace("%", "").split()[0])


def derived_ratios(case: dict) -> dict:
    """Ratios an analyst would compute before believing a GenAI business case."""
    f = case["facts"]
    tasks, people = _num(f["Work orders per year"]), _num(f["Planners using the tool"])
    saved, review = _num(f["Minutes saved per work order before review"]), _num(f["Review minutes per draft"])
    error, cash = _num(f["Share of drafts with an error reaching the floor"]), _num(f["Share of saved hours that becomes cash"])
    return {
        "Work orders per planner per working day (250 days)": f"{tasks / people / 250:,.1f}",
        "Minutes saved as a share of the manual task time": f"{saved / case['manual_minutes']:.0%} of {case['manual_minutes']} minutes",
        "Review time as a share of time saved": f"{review / saved:.0%}" if saved else "n/a",
        "Errors reaching the floor vs wrong drafts in the pilot": f"{error:.1f}% vs {case['pilot_wrong_rate']:.0%}",
        "Share of saved time counted as cash": f"{cash:.0f}%",
        "Setup hours for a tool that must be connected to the maintenance system": f["Setup hours"],
    }


FACT_NAMES = {
    "tasks_per_year": "Work orders per year",
    "minutes_saved_per_task": "Minutes saved per work order before review",
    "review_minutes_per_task": "Review minutes per draft",
    "hourly_cost": "Loaded hourly cost",
    "people": "Planners using the tool",
    "setup_hours": "Setup hours",
    "training_hours_per_person": "Training hours per person",
    "error_rate": "Share of drafts with an error reaching the floor",
    "share_redeployed": "Share of saved hours that becomes cash",
    "model_cost_per_task": "Model cost per draft",
}
HOURLY_BAND = (20, 150)          # loaded $/hour for non-executive staff
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def plausibility_checks(case: dict) -> list[dict]:
    """One generic rule per assumption (v3 red team). Returns flags, most severe first."""
    v = {k: _num(case["facts"][label]) for k, label in FACT_NAMES.items()}
    manual, wrong = case["manual_minutes"], case["pilot_wrong_rate"]
    hours_a_day = v["tasks_per_year"] * manual / max(v["people"], 1) / 250 / 60
    flags = []

    def flag(assumption, severity, issue):
        flags.append({"assumption": assumption, "severity": severity, "issue": issue})

    if v["people"] <= 0:
        flag("people", "high", "No one is listed as using the tool.")
    if hours_a_day > 4:
        flag("tasks_per_year", "high", f"{v['tasks_per_year']:,.0f} tasks at {manual} minutes each would take "
             f"{hours_a_day:.1f} hours of every working day for each of {v['people']:.0f} people.")
    if v["minutes_saved_per_task"] > manual:
        flag("minutes_saved_per_task", "high", f"Saves {v['minutes_saved_per_task']:.0f} minutes on a task that takes {manual} minutes.")
    elif v["minutes_saved_per_task"] > 0.8 * manual:
        flag("minutes_saved_per_task", "medium", f"Saves {v['minutes_saved_per_task']:.0f} of {manual} minutes, leaving almost no human work.")
    if v["review_minutes_per_task"] == 0 and wrong > 0.05:
        flag("review_minutes_per_task", "high", f"No review time, but {wrong:.0%} of pilot drafts had the wrong cause.")
    elif v["minutes_saved_per_task"] and v["review_minutes_per_task"] < 0.1 * v["minutes_saved_per_task"]:
        flag("review_minutes_per_task", "medium", f"Review is under 10% of the time saved ({v['review_minutes_per_task']:.0f} minutes).")
    if not HOURLY_BAND[0] <= v["hourly_cost"] <= HOURLY_BAND[1]:
        flag("hourly_cost", "high", f"${v['hourly_cost']:,.0f} an hour is outside the usual ${HOURLY_BAND[0]}-${HOURLY_BAND[1]} loaded cost.")
    if v["setup_hours"] < 8:
        flag("setup_hours", "high", f"{v['setup_hours']:.0f} setup hours for a tool that must be connected to another system.")
    if v["training_hours_per_person"] < 1:
        flag("training_hours_per_person", "medium", f"{v['training_hours_per_person']:.0f} training hours for a new tool.")
    if v["error_rate"] == 0:
        flag("error_rate", "high", f"Assumes no errors reach the floor, but {wrong:.0%} of pilot drafts were wrong.")
    if v["share_redeployed"] > 80:
        flag("share_redeployed", "high", f"Counts {v['share_redeployed']:.0f}% of saved time as cash; that needs fewer hires or less overtime.")
    if v["model_cost_per_task"] <= 0:
        flag("model_cost_per_task", "medium", "Assumes the model is free to run.")
    return sorted(flags, key=lambda f: SEVERITY_ORDER[f["severity"]])


def facts_text(facts: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())


def user_message(agent: str, case: dict, version: int = 1) -> str:
    if agent == "intake":
        return case["description"]
    text = f"Business case: {case['title']}\nContext: {case['context']}\nFacts:\n{facts_text(case['facts'])}"
    if agent == "redteam" and version == 2:
        text += f"\nRatios computed from the inputs:\n{facts_text(derived_ratios(case))}"
    if agent == "redteam" and version == 3:
        flags = plausibility_checks(case)
        text += "\nFlags:\n" + ("\n".join(f"- {f['assumption']} ({f['severity']}): {f['issue']}" for f in flags) or "none")
    return text


# ---------------------------------------------------------------- v2 post-processing (code, not the model)
PER_YEAR = {"year": 1, "month": 12, "week": 52}


def finalize_intake(reply: str, description: str) -> str:
    """Keep only numbers whose quote appears in the text, then convert units in code."""
    d = parse(reply) or {}
    text = description.lower()

    def grounded(key):
        item = d.get(key) or {}
        quote, value = item.get("quote"), item.get("value")
        ok = isinstance(value, (int, float)) and isinstance(quote, str) and quote.strip().lower() in text
        return (value, item) if ok else (None, item)

    vol, vol_item = grounded("volume")
    days, _ = grounded("working_days")
    per = (vol_item.get("per") or "").lower()
    tasks = None
    if vol is not None:
        tasks = vol * (PER_YEAR.get(per) or (days or 250 if per == "day" else 0)) or None

    def minutes(key):
        value, item = grounded(key)
        return None if value is None else value * (60 if (item.get("unit") or "").lower().startswith("hour") else 1)

    out = {"use_case": d.get("use_case"), "tasks_per_year": tasks, "minutes_saved_per_task": minutes("minutes_saved"),
           "review_minutes_per_task": minutes("review_time"), "hourly_cost": grounded("hourly_cost")[0],
           "people": grounded("people")[0], "questions": d.get("questions", [])}
    return json.dumps(out)


PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def finalize_memo(reply: str, facts: dict) -> str:
    """Swap {fact name} placeholders for the exact figures; unknown names are left visible as [unknown: ...]."""
    return PLACEHOLDER.sub(lambda m: str(facts.get(m.group(1).strip(), f"[unknown: {m.group(1)}]")), reply)


def finalize_redteam(reply: str) -> str:
    """Drop objections whose assumption name is not on the list."""
    d = parse(reply) or {}
    objections = [o for o in d.get("objections", []) if isinstance(o, dict) and o.get("assumption") in ASSUMPTIONS]
    return json.dumps({**d, "objections": objections, "dropped": len(d.get("objections", [])) - len(objections)})


def finalize_redteam_v3(reply: str, case: dict) -> str:
    """Code owns which assumptions are flagged, their severity and the verdict; the model adds why + test."""
    d = parse(reply) or {}
    written = {o.get("assumption"): o for o in d.get("objections", []) if isinstance(o, dict)}
    flags = plausibility_checks(case)
    objections = [{**f, "why": written.get(f["assumption"], {}).get("why"),
                   "test": written.get(f["assumption"], {}).get("test")} for f in flags]
    severities = {f["severity"] for f in flags}
    verdict = "send back" if "high" in severities else "approve with conditions" if severities else "approve"
    return json.dumps({"objections": objections, "verdict": verdict,
                       "explained": sum(bool(o["why"] and o["test"]) for o in objections),
                       "dropped": len([a for a in written if a not in {f["assumption"] for f in flags}])})


def finalize(agent: str, reply: str, case: dict, version: int = 2) -> str:
    if agent == "intake":
        return finalize_intake(reply, case["description"])
    if agent == "memo":
        return finalize_memo(reply, case["facts"])
    return finalize_redteam_v3(reply, case) if version == 3 else finalize_redteam(reply)


# ---------------------------------------------------------------- graders
def grade_intake(reply: str, expected: dict) -> dict:
    d = parse(reply) or {}
    fields = {}
    for f in INTAKE_FIELDS:
        got, want = d.get(f), expected.get(f)
        if want is None:
            fields[f] = got is None                                   # must not invent
        else:
            fields[f] = isinstance(got, (int, float)) and abs(got - want) <= 0.01 * abs(want)
    return {"valid_json": bool(d), "fields": fields, "score": sum(fields.values()) / len(fields),
            "invented": [f for f in INTAKE_FIELDS if expected.get(f) is None and d.get(f) is not None]}


NUM = re.compile(r"(\$?)(\d[\d,]*\.?\d*)\s*(k|K|M|million|%)?")


def numbers_in(text: str) -> list[float]:
    """Numbers a reader would take as figures: money, percentages and any number above 12."""
    out = []
    for dollar, num, suffix in NUM.findall(text):
        try:
            x = float(num.replace(",", ""))
        except ValueError:
            continue
        if suffix in ("k", "K"):
            x *= 1e3
        elif suffix in ("M", "million"):
            x *= 1e6
        if dollar or suffix == "%" or x > 12:
            if not (1990 <= x <= 2100 and not dollar and not suffix):  # skip years
                out.append(x)
    return out


def grade_memo(reply: str, facts: dict, context: str = "", tolerance: float = 0.05) -> dict:
    allowed = numbers_in(" ".join(str(v) for v in facts.values()) + " " + context)
    used = numbers_in(reply)
    invented = [x for x in used if not any(abs(x - a) <= tolerance * max(abs(a), 1) for a in allowed)]
    headings = ["## Recommendation", "## Value", "## Risks", "## Next step"]
    return {"numbers_used": len(used), "invented": invented,
            "faithful": not invented, "has_sections": all(h in reply for h in headings),
            "words": len(reply.split())}


def grade_redteam(reply: str, planted: str | None) -> dict:
    d = parse(reply) or {}
    objections = d.get("objections", []) if isinstance(d.get("objections"), list) else []
    keys = [o.get("assumption") for o in objections if isinstance(o, dict)]
    serious = [o.get("assumption") for o in objections if isinstance(o, dict) and o.get("severity") in ("high", "medium")]
    return {"valid_json": bool(d), "flagged": keys, "verdict": d.get("verdict"),
            "high": sum(isinstance(o, dict) and o.get("severity") == "high" for o in objections),
            "caught": planted in serious if planted else None,
            "caught_first": bool(keys) and keys[0] == planted if planted else None}


def cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1000 * PRICE_PER_1K["input"] + output_tokens / 1000 * PRICE_PER_1K["output"]


def run_live(agent: str, case: dict, version: int = 3, region: str = "us-east-2", profile: str | None = None) -> dict:
    """Call Bedrock directly (local use only; needs AWS credentials and boto3)."""
    import boto3                                       # optional dependency: not needed to view saved results
    client = boto3.Session(profile_name=profile, region_name=region).client("bedrock-runtime")
    resp = client.converse(
        modelId=MODEL_ID, system=[{"text": PROMPTS[version][agent]}],
        messages=[{"role": "user", "content": [{"text": user_message(agent, case, version)}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS[agent], "temperature": 0},
    )
    reply = resp["output"]["message"]["content"][0]["text"]
    return {"reply": reply, "final": finalize(agent, reply, case, version) if version > 1 else reply,
            "input_tokens": resp["usage"]["inputTokens"], "output_tokens": resp["usage"]["outputTokens"],
            "latency_ms": resp["metrics"]["latencyMs"]}
