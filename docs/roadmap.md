# Product roadmap: AI Use-Case ROI Navigator

*Last updated October 2026 · Owner: Chaitali Shekar · Format: Now / Next / Later*

## Vision

Help an AI provider **match the right AI to each client and prove it keeps paying**, from the first conversation to a year after launch,
with numbers a CFO will sign and a person in charge of every decision.

**Who it is for:** the AI business analyst or value engineer who builds the case; the sponsor (ops director, CFO) who
funds it; the plant or business team who live with it after launch.

**How we will know it works** (product metrics):

| Outcome | Metric | Today |
|---|---|---|
| Decisions get made faster | Time from use-case idea to a signed business case | Same session (demo data) |
| The numbers are trusted | Share of business-case assumptions with a source (invoice, system data, pilot) | Model and pilot measured; plant costs illustrative |
| Value is tracked, not assumed | Months between a problem starting and it being flagged | 2 months (drift flagged in month 11 of the simulated rollout) |
| GenAI is measured before it is trusted | Agent tasks with an automatic grader | 3 of 3 agents, 19 cases |

## Status overview

7 steps shipped (v1), including client matching. 5 items planned Next, 5 Later. Nothing blocked. One known risk: the agent tests are not blind (see N1).

## How items were prioritised

The same value × feasibility scoring the app uses on step 1, each scored 1–5:

| Item | Value | Feasibility | Priority | Quadrant |
|---|---|---|---|---|
| N1 Blind test set for the agents | 4 | 5 | 20 | Quick win |
| N2 Scheduled pipeline with alerts | 4 | 4 | 16 | Quick win |
| N3 Value models for every solution | 5 | 3 | 15 | Big bet, start small |
| N5 Track AI running costs, not just savings | 4 | 4 | 16 | Quick win |
| N4 Usage analytics on the live app | 3 | 5 | 15 | Quick win |
| L1 Portfolio view across many AI projects | 5 | 2 | 10 | Big bet |
| L2 Governance pack | 4 | 3 | 12 | Big bet |
| L3 Connect to a real maintenance system | 5 | 2 | 10 | Big bet |
| L4 LLM-as-judge, calibrated by people | 3 | 3 | 9 | Fill-in |
| L5 Sign-in and saved cases | 3 | 2 | 6 | Later, only with users |

## Now: shipped (v1)

| Item | What it does | Status |
|---|---|---|
| Match AI to client | A library of 12 solutions matched to the client's problem, data, risk appetite and urgency, with reasons | **Done** |
| Prove the value | Real model on the AI4I 2020 dataset, tuned for dollars; low / base / high case and the biggest risk | **Done** |
| Check readiness | 10 questions, 5 areas, Go / Fix first / Not ready with actions | **Done** |
| Business case | One-page summary for finance, downloadable | **Done** |
| Track the value | Forecast vs actual, adoption and drift alerts, gap explained by cause; S3 → Databricks bronze / silver / gold pipeline | **Done** |
| GenAI pilot | Bedrock work-order writer measured on 30 alerts; ROI across five cost categories | **Done** |
| AI analyst agents | Intake, memo and red-team agents graded v1 → v3 | **Done** |
| Client-first UX | Overview with the impact up front; every page answer first, analysis on demand | **Done** |

## Next: 1–3 months

| # | Item | Why | Done when | Depends on |
|---|---|---|---|---|
| N1 | **Blind test set for the agents** | The red-team checklist was written knowing the planted flaws, so 6 of 6 overstates it | 10 new cases written by someone else, scored once, result published even if worse | A second person to write cases |
| N2 | **Scheduled pipeline with alerts** | Tracking only matters if it runs without anyone remembering | Monthly Databricks job unpaused; drift or low adoption sends an email (SNS); a run log page in the app | Databricks job (exists, paused) |
| N3 | **Value models for every solution** | Step 1 now recommends the right AI for any client, but only predictive maintenance has a full dollar model behind it | The hospital's and retailer's matches get a business case from the generic time-saved / error-avoided calculator; the intake agent fills it from the client's own description | N1, so intake is trusted |
| N5 | **Track AI running costs, not just savings** | Running cost is a fixed $60k assumption today. Model fees, managed-platform premiums and idle compute drift after launch, and nobody revisits them | Monthly actual running cost from the AWS cost data next to the business-case assumption, with an alert when it drifts 20%+ | AWS Cost Explorer access |
| N4 | **Usage analytics on the live app** | Learn which steps people open and where they stop | Page views and "See the analysis" opens counted, no personal data | Public URL confirmed |

## Later: 3–6+ months (directional)

| # | Item | Why |
|---|---|---|
| L1 | **Portfolio view** | Leaders fund many AI projects at once; one screen ranking them by value delivered vs promised |
| L2 | **Governance pack** | Each business case also produces a model card and a NIST AI RMF / EU AI Act checklist, so risk review is not a separate project |
| L3 | **Connect to a real maintenance system** | Replace the simulated rollout with a live export (SAP PM, IBM Maximo or a CMMS CSV) at a pilot site |
| L4 | **LLM-as-judge, calibrated by people** | Grade open-ended agent output (memo quality) where exact-match graders cannot, checked against human ratings |
| L5 | **Sign-in and saved cases** | Only once there are repeat users; until then a download is enough |

## Not doing (and why)

- **Fine-tuning our own model.** The pilot showed accuracy problems came from asking the model to judge numbers; code fixed that for less.
- **Multi-Region deployment.** One Region (us-east-2) is enough for a demo and keeps cost and data location simple.
- **Real-time streaming.** Maintenance value is reviewed monthly; streaming adds cost without changing a decision.

## Risks and dependencies

| Risk | Effect | Mitigation |
|---|---|---|
| Agent results look better than they are (N1) | Credibility with a technical reviewer | Publish the blind result whatever it is; keep the honest-limits note in the app |
| Illustrative costs are mistaken for real ones | A reader over-trusts the dollar figures | "What is real" note on the overview; costs editable on every page |
| Databricks Free Edition limits (no S3 external location) | Sync step is a workaround | Documented; switch to an external location on a paid workspace |
| AWS spend | Unexpected bill | Live calls only with local credentials; public app uses saved results |
