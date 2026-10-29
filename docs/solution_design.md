# The ROI Compass: Solution Design

*Independent concept project.*

---

## 1. Introduction
The ROI Compass is a browser-based tool for the work of an **AI business systems analyst**: taking a business need through to an AI that is live and paying off. It helps the analyst and an operations leader:
1. pick the AI that fits the business need,
2. prove its value in dollars using a real model,
3. check whether the team is ready, and get finance to sign off, and
4. after launch, check every month that it delivers what was promised.

**Intended audience:** operations leaders evaluating AI, and the AI business systems analysts who work with them.

Process maps, the integration spec, data dictionary, acceptance tests and traceability matrix are in [requirements.md](requirements.md).

**Problem:** an AI provider sells a library of AI tools; its clients ask which one to buy first and whether it will pay off. Step 1 compares the library for one client in dollars. **Deep dive:** predictive maintenance for an auto-parts plant, tested on the public AI4I 2020 dataset.

---

## 2. Business Problem and Business Value

### 2.1 Business problem
Operations leaders at mid-size manufacturers face three gaps when adopting AI:

| Gap | What happens today | Consequence |
|---|---|---|
| **Choosing** | Many AI ideas, no objective way to compare them | Projects are picked on hype, not value |
| **Proving value** | Models are judged on accuracy, not on cost of errors | Finance can't approve a business case |
| **Going live** | No check of data, workflow fit or user trust before launch | Pilots stall and never reach production |

### 2.2 Business value
| Value driver | How the Navigator delivers it | Measure |
|---|---|---|
| Faster decisions | Prices every AI tool that fits the client in dollars, with ranges and evidence, in one view | Time from a vendor's pitch to a priced shortlist |
| Credible business case | Converts model results into annual savings, ROI and payback | Annual net savings ($), payback (months) |
| Higher pilot-to-production rate | Readiness scorecard flags blockers before launch | Readiness verdict with top 3 actions |
| Faster time to value | Compares a custom build with a ready-made solution | First-year value ($) by months to deploy |

### 2.3 Personas
**P1 · Dana, Plant Operations Director (primary user)**
- Owns uptime, cost per unit and maintenance budget.
- Not a data scientist. Thinks in downtime hours and dollars.
- Needs a business case she can take to her CFO.

**P2 · AI Business Analyst (supporting user)**
- Advises customers like Dana on which AI to adopt.
- Runs the session, tailors assumptions, explains the results.

---

## 3. Technical Overview

### 3.1 Use cases
| ID | Use case | Primary actor | Trigger | Outcome |
|---|---|---|---|---|
| UC-1 | Compare the AI library | Dana, with the AI BSA | A provider pitches several AI tools: "which should we buy first?" | Every fitting tool priced in dollars with ranges and evidence; a first pick, with reasons |
| UC-2 | Prove value of predictive maintenance | Dana, with the AI BSA | An option rated "Start now" is picked | Annual $ saved, ROI and payback from a tested model |
| UC-3 | Check production readiness | Dana | Business case is positive | Go / Fix first / Not ready, with top actions |
| UC-4 | Share the business case | Dana | Ready to request funding | One-page summary for the CFO |

**UC-2 detail: Prove value of predictive maintenance**
- **Preconditions:** a trained model and test results are loaded; default cost assumptions are set.
- **Main flow:**
  1. Dana enters her plant's costs (missed failure, false alarm, deployment, running cost).
  2. The tool shows annual net savings, ROI and payback against a "no AI" baseline.
  3. The AI BSA moves the alert-threshold slider; failures caught, false alarms and dollars update.
  4. The tool highlights the dollar-optimal threshold next to the accuracy-optimal one.
  5. Dana sets months to deploy and sees first-year value.
- **Alternate flow:** if net savings are negative at every threshold, the tool recommends **do not deploy** and shows which cost drives the loss.
- **Postcondition:** a recommended threshold and a dollar business case are ready for UC-3 and UC-4.

### 3.2a Mapping to the AI lifecycle
Enterprise AI services are commonly organised around four lifecycle stages: **advisory → implementation → adoption → managed services**, with the goal of moving customers **from pilot to production**. The epics below follow those stages, and every story is tagged with the lifecycle stage and a MoSCoW priority (Must / Should / Could / Won't for v1).

| Lifecycle stage | Customer question | Epics | Built in v1 |
|---|---|---|---|
| **Advisory** | Which AI fits this client, and is it worth it? | Epic 1 Match AI to the client, Epic 2 Prove value | Done |
| **Implementation** | Are we ready, and how do we deploy safely? | Epic 3 Readiness, Epic 5 Responsible AI & deployment | Partly |
| **Adoption** | Will people use it, and can we prove it to finance? | Epic 4 Share, Epic 6 Adoption & value tracking | Done |
| **Managed services** | Is it still delivering value after launch? | Epic 6 Adoption & value tracking | Partly |

### 3.2 Functional requirements (user stories and acceptance criteria)

#### Epic 1 · Compare the AI library for the client (UC-1)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-1.1 | As an ops director, I want every AI tool that fits my industry priced for my business in dollars, so that I know which to buy first. | Only the client's industry; value = problem cost × published improvement; 3-year net value, payback, ROI and break-even per tool; ranking matches a hand calculation (unit test). | Must (done) |
| US-1.2 | As an ops director, I want to enter my own problem costs, data quality and number of sites, so that the numbers are mine, not an example. | Editable under Show the details; ranking updates at once. | Must (done) |
| US-1.3 | As a finance partner, I want every number shown as a range with its source, so that I can judge how far to trust it. | Low / expected / high per tool; evidence label and source link on every row. | Must (done) |
| US-1.4 | As an ops director, I want a next step for the top pick, so that I know what to do with it. | Links to the full business case when it is predictive maintenance; recommends a pilot first when the top pick rests on weak evidence. | Should (done) |
| US-1.5 | As an AI business systems analyst, I want missing data to cost extra, so that tools needing perfect data do not look unfairly good. | Each point the client's data falls short adds 20% to setup and 2 months to go live (unit test). | Must (done) |
| US-1.6 | As a business sponsor, I want to see why a tool comes first and when the runner-up is close, so that I can challenge it. | The answer names the 3-year value, payback and evidence; when the runner-up is within 25%, it says payback and evidence decide. | Must (done) |
| US-1.7 | As a finance partner, I want to see whether the order holds on cautious numbers, so that a tool cannot win on optimistic figures alone. | Expected / cautious switch; rank changes listed (unit test: chip fab). | Must (done) |

#### Epic 2 · Prove value (UC-2)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-2.1 | As an ops director, I want to enter my own costs, so that savings reflect my plant. | Inputs: breakdowns a year, cost of a breakdown, planned-repair cost, inspection cost per false alarm, yearly running cost, setup cost, months to go live. Defaults provided; savings update on every change. | Must (done) |
| US-2.2 | As an ops director, I want annual savings, ROI and payback, so that I can take a business case to my CFO. | Compared against "no AI" (all failures missed); shows net savings, ROI % and payback months; updates on every input change.  | Must (done) |
| US-2.3 | As an AI BSA, I want to adjust the alert threshold and see the dollar effect, so that I can show accuracy is not the same as value. | Slider; shows recall, precision, caught, missed, false alarms, net $; marks the break-even level (set from the costs) and the usual 50% cut-off. | Should (done) |
| US-2.4 | As an ops director, I want results in plain language, so that I don't need ML knowledge. | e.g. "Catches 8 of 10 failures. About 1 in 3 alerts is a false alarm."  | Should (done) |
| US-2.5 | As an AI BSA, I want the model tested on unseen data, so that the numbers are trustworthy. | Train/test split; all reported metrics come from the test set.  | Must (done) |
| US-2.6 | As an ops director, I want to compare a custom build with a ready-made solution, so that I see how deployment time changes first-year value. | Input months to deploy; first-year value = annual savings × (12 − months) ÷ 12.  | Could (done) |
| US-2.7 | As a CFO, I want a range instead of a single number, so that I know how much the savings could vary. | Savings, payback, ROI, caught, missed and false alarms each show a 90% range from a bootstrap of the test set; the range always contains the point estimate.  | Must (done) |
| US-2.8 | As an ops director, I want low / base / high cases and to see which cost assumption matters most, so that we firm up the right numbers before committing. | Editable low and high value for every cost input; table of low / base / high annual savings, 3-year value, ROI and payback; tornado chart sorted by impact; warning if the low case never pays back within 3 years.  | Should (done) |

#### Epic 3 · Readiness (UC-3)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-3.1 | As an ops director, I want a short readiness checklist, so that I know if we're ready to go live. | ≤ 10 questions rated 1–5 across 5 areas: data, fits how people work, clear payoff, people trust it, clear ownership. Score per dimension.  | Must (done) |
| US-3.2 | As an ops director, I want a verdict and top actions, so that I know what to fix before launch. | **Not ready:** average < 2.5. **Fix first:** any area < 2.5, or average below 3.5. **Go:** average ≥ 3.5 and no area < 2.5. Lists the weakest areas below 3.5 with an action each. | Must (done) |

#### Epic 4 · Share (UC-4)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-4.1 | As an ops director, I want a one-page summary, so that I can share it with my CFO. | Two-page PDF: recommendation, decision rule, options considered, annual $ value with range, payback, readiness verdict, risks and key assumptions; downloadable. | Must (done) |


#### Epic 5 · Responsible AI & deployment (Implementation)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-5.1 | As an ops director, I want to see why the model flagged a machine, so that technicians trust and act on alerts. | Each alert shows its top 3 drivers (e.g. tool wear, torque) in plain language. | Should (v2) |
| US-5.2 | As an IT lead, I want to choose whether the model runs on-premises, at the edge or in the cloud, so that sensitive plant data stays where our policy requires. | Deployment option is selectable; running cost and data-location note update with the choice. | Could (v2) |
| US-5.3 | As a governance owner, I want an accountable owner and a review trigger for model performance, so that the AI stays safe and accurate after launch. | Readiness check includes owner and monitoring questions; verdict flags gaps. | Must (done) (covered by US-3.1) |

#### Epic 6 · Adoption & value tracking (Adoption → Managed services)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-6.1 | As an ops director, I want to compare predicted savings with actual results after go-live, so that I can prove the value to finance. | Upload monthly actuals (failures, downtime, false alarms); chart shows predicted vs actual savings. | Must (done) (step 5; gap split into volume, recall, adoption and false alarms) |
| US-6.2 | As an AI business analyst, I want to track adoption (share of alerts acted on), so that I can spot when users stop trusting the AI. | Adoption rate shown monthly; alert if it drops below 70%. | Should (done) |
| US-6.3 | As an AI business analyst, I want to be warned when model accuracy drifts, so that we retrain before value drops. | Alert when recall falls 10 points below launch level. | Could (done) (3-month rolling recall, since ~3 failures a month is too few to judge one month alone) |
| US-6.4 | As an ops director, I want the tool to suggest the next AI use case once the first is live, so that we expand value step by step. | After a "Go" verdict, the summary lists the next 2 "Start now" options from Epic 1. | Could (v2) |

#### Epic 7 · GenAI pilot (Advisory → Implementation)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-7.1 | As an AI business analyst, I want to count every cost of a GenAI assistant, not just the subscription, so that the ROI isn't overstated. | Five cost categories (usage, setup, review, training, errors); reproduces the Forbes worked example (162%). | Must (done) |
| US-7.2 | As a finance partner, I want to see cash value separately from freed-up time, so that I only bank savings that reach the P&L. | Share of saved hours that becomes cash is an input; cash ROI shown next to ROI. | Must (done) |
| US-7.3 | As an AI business analyst, I want to measure cost, speed and accuracy in a real pilot before committing, so that the business case uses evidence instead of guesses. | 30 real alerts run on Amazon Bedrock; tokens, latency and cost recorded per call; drafts graded against the known cause. | Must (done) |
| US-7.4 | As a maintenance manager, I want the AI's diagnosis to be reliable, so that technicians don't act on a wrong cause. | Diagnosis moved to a rule check; model only writes the work order; right cause on all real failures in the pilot. | Must (done) |
| US-7.5 | As an AI business analyst, I want to know the review time at which the assistant stops paying, so that I know what to measure in production. | Break-even review minutes per draft shown. | Should (done) |

#### Epic 8 · Analyst agents and evals (Implementation)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-8.1 | As an ops manager, I want to describe a use case in plain English and get calculator inputs, so that I don't need to know the ROI model. | Intake agent extracts tasks/year, minutes saved, review minutes, hourly cost and people; converts units in code; anything not stated stays blank with a question. | Must (done) |
| US-8.2 | As a CFO, I want a one-page memo where every number matches the business case, so that I can trust what I sign. | Memo agent writes with `{fact}` placeholders that code fills in; grader finds no number that isn't in the facts. | Must (done) |
| US-8.3 | As a CFO, I want a skeptical review that finds the weakest assumption, so that inflated business cases are sent back before approval. | Planted flaw in 6 of 6 cases raised as high/medium and listed first; no high-severity alarm on a sound case; each flag has a 30-day test. | Must (done) |
| US-8.4 | As an AI business analyst, I want every agent graded automatically on cases with known answers, so that prompt changes are measured, not judged by eye. | `python -m agents.grade` prints a v1 / v2 / v3 scorecard from saved runs; graders covered by unit tests. | Must (done) |
| US-8.5 | As the app owner, I want the public demo to show saved results and run live only where AWS credentials exist, so that visitors can't spend my credits. | Live intake only when an AWS profile is available; otherwise saved results with an explanation. | Should (done) |

#### Epic 9 · Data pipeline (Implementation → Managed services)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-9.1 | As a plant IT lead, I want to drop each month's actuals file in one secure place, so that nobody emails spreadsheets around. | Private S3 bucket in us-east-2: public access blocked, encrypted, versioned, HTTPS only, access logged to CloudWatch. | Must (done) |
| US-9.2 | As an AI business analyst, I want bad files stopped before they reach the numbers, so that finance never sees impossible savings. | Silver layer applies the Track value checks; failing rows go to a quarantine table with the reason; a planted bad file is quarantined. | Must (done) |
| US-9.3 | As a finance partner, I want the pipeline's numbers to match the app exactly, so that there is one version of the truth. | Gold table (Databricks SQL) equals `src/tracking.py` (pandas) for every month, to the cent, including alert flags; checked by a unit test and shown in the app. | Must (done) |
| US-9.4 | As a data engineer, I want re-runs to be safe, so that a retry never double-counts a month. | `COPY INTO` skips files already loaded (second run loads 0 rows); a resent file for a month replaces the earlier one in silver. | Should (done) |

### 3.3 Non-functional requirements
| Category | Requirement |
|---|---|
| Access | Runs in a web browser; no login |
| Performance | Page loads in under 5 seconds; recalculations in under 1 second |
| Transparency | Every assumption is labeled and editable |
| Trust | Model metrics reported on held-out test data only |
| Branding | Independent project; no company branding |
| Clarity | A first-time visitor understands the point in under a minute: an overview that states the problem and the money at stake, with an interactive tour of the steps, and every page shows the client view first (question, answer, three numbers, one chart, next steps) with the analysis (assumptions, method, data) one click away |
| Navigation | Pages grouped by stage (Plan the project, After launch, Generative AI) with their own links, Back / Next buttons on every page, and every page works on its own with default inputs |
| Motion | Short fade-in when a page opens; switched off for visitors who ask their system for reduced motion |

---

## 4. Architectural Overview
```
User (browser)
   │
   ▼
Case-study website (site/, static) ──► home: the problem, 5 options, the 7-phase staircase, key decisions
   │  one page per phase, each with "Show the details" (full working; live inputs on Business case and Readiness)
   │  numbers come from site/export_data.py, which calls the same src/ functions as the tests
   ▼
Hosted on GitHub Pages (gh-pages branch)

Analyst workbench (app.py, Streamlit, run locally) ──► the same tools plus live Bedrock runs (needs AWS credentials)
```

Value-tracking data pipeline (Epic 9):
```
Plant maintenance system
   │ monthly actuals CSV
   ▼
S3 landing zone  s3://ai-roi-data-…-us-east-2-an/landing/actuals/month=YYYY-MM/   (private, encrypted, versioned, logged)
   │ pipeline/run.py sync  (Free Edition cannot mount S3; paid tier: Unity Catalog external location)
   ▼
Databricks volume workspace.ai_roi.landing
   ▼  job "ai-roi-value-pipeline" (serverless, monthly schedule ready)
bronze_actuals ──► silver_actuals ──► gold_tracked_value ──► export ──► app page 5  +  S3 curated/
               └─► quarantine_actuals (row + reason)
```

---

## 5. Component Model
| Component | Responsibility | File |
|---|---|---|
| AI library | Each AI tool per industry: published improvement range, evidence and source, provider price, data needed | `data/ai_library.csv` |
| Library ranking | Prices every tool for one client, ranks on expected or cautious numbers, flags close calls and rank changes | `src/library.py` |
| Model training | Train and test a failure-prediction model on AI4I 2020 | `src/model.py` |
| Cost engine | Converts caught / missed / false alarms into dollars, ROI and payback; bootstrap ranges and sensitivity cases | `src/value.py` |
| Readiness engine | Scores dimensions and applies verdict rules | `src/readiness.py` |
| GenAI cost engine | Time saved minus five cost categories; cash vs capacity; break-even review time | `src/genai.py` |
| Work-order pilot | Picks 30 alerts, builds prompts, rule-check diagnosis, grades drafts; runs on Amazon Bedrock (Nova Micro, us-east-2) | `src/workorder.py`, `pilot/` |
| Analyst agents | Intake, memo and red-team prompts (v1-v3), code post-processing, plausibility checklist, graders; Amazon Nova Lite on Bedrock | `src/agents.py`, `agents/` |
| Data pipeline | S3 landing, sync to Databricks, bronze/silver/gold notebook, job, export back to the app and S3 | `pipeline/run.py`, `pipeline/notebook.py`, `data/pipeline/` |
| Value tracking | Forecast vs actual savings, adoption and drift alerts, gap breakdown, simulated sample rollout | `src/tracking.py`, `data/sample_actuals.csv` |
| User interface | Static website: 7 phase pages, answer first, "Show the details" with the full working and live inputs; local Streamlit workbench | `site/`, `app.py` |

---

## 6. Operational Model
How a typical 30-minute session runs:

| Step | Who | Activity | Output |
|---|---|---|---|
| 1. Discover | AI BSA with Dana | Agree the business's main problem, data quality, tolerance for AI mistakes and urgency | Business profile |
| 2. Pick | AI BSA with Dana | Review the recommended solution, its reasons and the ranked library | Recommended AI and the blocker to fix, if any |
| 3. Prove value | AI BSA with Dana | Enter plant costs; set the warning level | Dollar business case |
| 4. Check readiness | Dana | Complete the checklist | Verdict and top 3 actions |
| 5. Share | Dana | Download the one-page summary | CFO-ready business case |

### 6.1 Why value falls short: fishbone

After the plant-wide rollout the AI beat the approved plan but banked 70% of the model's estimate. Before measuring anything, these are all the causes that could explain a shortfall, grouped the usual six ways. The right column says which ones the app measures today.

```
  PEOPLE                     PROCESS                    MODEL
  crew ignores alerts        alert not in the           recall drifts as
  no trust in the AI         work-order system          machines and tools age
  planner turnover           no owner for retraining    threshold set once, never tuned
          \                          |                          /
           \                         |                         /
            >───────────────── VALUE BELOW FORECAST (70%) ◄────
           /                         |                         \
          /                          |                          \
  DATA                       ENVIRONMENT                MEASUREMENT
  late or wrong monthly      fewer breakdowns than      cost per failure never
  files from the plant       assumed (good news)        checked against invoices
  sensor outages             production volume drops    running cost assumed, not tracked
```

| Category | Cause | Measured in the app? | Effect in the simulated year |
|---|---|---|---|
| People | Crew ignores correct alerts | Yes: adoption alert below 70% | −$63k |
| Model | Recall drifts below launch | Yes: drift alert, 10+ points below launch | −$46k |
| Environment | Fewer breakdowns than assumed | Yes: volume effect in the gap breakdown | −$91k |
| Data | Bad or late monthly file | Yes: pipeline quarantine with the reason | 1 file quarantined |
| Process | No owner for retraining or threshold | Partly: readiness check asks; RACI below assigns | n/a |
| Measurement | Cost per failure unverified | No: counts cannot check dollars; flagged as the top assumption to agree with finance | n/a |
| Measurement | AI running cost drifts from the $60k assumption | No: on the roadmap (N5) | n/a |

### 6.2 Who owns what after launch: RACI

R = does the work, A = accountable (one per row), C = consulted, I = informed.

| Activity | AI BSA | Ops director | Maintenance manager | Data / IT lead | Finance partner |
|---|---|---|---|---|---|
| Pick the AI for the business need | R | A | C | C | I |
| Agree the cost assumptions | R | C | C | I | A |
| Readiness actions before launch | C | A | R | R | I |
| Send monthly actuals file | I | I | C | A, R | I |
| Fix quarantined files | I | I | R | A | I |
| Review value vs forecast each month | R | A | C | I | C |
| Act on adoption alerts (crew workflow) | C | A | R | I | I |
| Retrain on drift alert | C | I | C | A, R | I |
| Revisit the business case each quarter | R | A | C | C | C |
| Approve renewal or expansion | C | A | I | I | C |

The last row is the point of the whole design: someone is accountable for going back to the original decision.

---

## 7. Deployment Considerations
- **Hosting:** GitHub Pages for the website (free, static, no server). The Streamlit workbench runs locally.
- **Data:** public AI4I 2020 dataset only; no customer data is stored.
- **Customer use:** for real customer data, run the app locally or inside the customer's own environment so data stays on-premises.
- **Dependencies:** listed in `requirements.txt` (Streamlit, pandas, Plotly, scikit-learn). The pipeline and live agents also need `boto3[crt]` and `databricks-sdk`; the website does not.
- **Pipeline:** S3 bucket in us-east-2 only (project Region). The website and the workbench read the exported gold table in `data/pipeline/`, so it needs no AWS or Databricks credentials. The job's monthly schedule is created paused; turn it on when real files arrive.
- **Cost:** S3 storage, requests and CloudWatch log delivery for these files are fractions of a cent a month; Databricks Free Edition serverless compute is free within its daily quota.

---

## 8. Solution Validation
Each acceptance criterion is tested before release. Results below; run `pytest` to repeat the automated tests.

| Test | Linked stories | Method | Result |
|---|---|---|---|
| Ranking matches manual calculation | US-1.1 | Compare app output to hand calculation | Passed (auto-parts plant, predictive maintenance: $1M × 35% − $60k = $290k a year; 3-year net $672k; payback 8.2 months) |
| Edited client figures update the ranking at once | US-1.2 | Manual UI test | Passed |
| Model metrics come from test data only | US-2.5 | Stratified 70/30 split; all metrics on 3,000 unseen records | Passed |
| Dollar-optimal threshold differs from accuracy-optimal | US-2.3 | Compare both on the test set (default costs) | Passed: break-even level 0.02 (set from the costs) vs the usual 0.5 cut-off; break-even saves $20k/year more ($658k vs $638k) |
| Verdict rules give the expected result for 3 sample checklists | US-3.2 | Unit tests (`tests/test_core.py`) | Passed |
| Dollar math matches hand calculation | US-2.2 | Unit test on a 4-record example | Passed |
| Uncertainty range contains the point estimate | US-2.7 | Unit test (`tests/test_core.py`) | Passed: default case $658k, range $609k–$703k |
| Scenarios are ordered low ≤ base ≤ high; tornado sorted by impact | US-2.8 | Unit test (`tests/test_core.py`) | Passed: low case $162k/year, 20.7-month payback |
| Actual savings match hand calculation; adoption and drift alerts fire on the right months | US-6.1, US-6.2, US-6.3 | Unit test on a 4-month example | Passed |
| Gap causes add up exactly to actual − forecast; actuals equal to the forecast give no gap | US-6.1 | Unit tests (5 simulated rollouts + exact-match case) | Passed: sample rollout $464k vs $658k model estimate, gap −$195k |
| Bad actuals files are rejected with a clear message | US-6.1 | Unit test (missing column, impossible counts) | Passed |
| GenAI ROI reproduces the Forbes bookkeeping example | US-7.1 | Unit test | Passed: cost $6,350, gain $16,640, ROI 162% |
| Cash + capacity = total gain; break-even review gives zero net value | US-7.2, US-7.5 | Unit tests | Passed |
| Pilot v1 (model diagnoses and writes) | US-7.3 | 30 Bedrock calls, graded against true cause | Failed: 33% right cause (42% on true failures, 0 of 6 false alarms); $0.000038 per draft, 0.9 s |
| Pilot v2 (rule check diagnoses, model writes) | US-7.4 | Same 30 alerts | Passed on these 30 synthetic examples: 100% right cause on true failures (83% overall), diagnosis kept in 30/30 drafts; 4 drafts mention PC parts, so review stays |
| Rule check matches the manual and every real failure in the pilot set | US-7.4 | Unit test | Passed |
| Agents v1 (model does everything) | US-8.1–8.3 | 17 Bedrock calls (Nova Lite), graded in code | Failed: Intake 42% fields right (arithmetic inside JSON); 1 of 2 memos invented a number; red team caught 3 of 6 planted flaws |
| Agents v2 (model quotes, code converts and fills in numbers) | US-8.1, US-8.2 | 19 calls, same graders | Passed: Intake 100%, nothing invented; 2 of 2 memos faithful. Failed: Red team still 2 of 6, even when given the ratios |
| Agents v3 red team (code checklist judges, model explains) | US-8.3 | 9 calls | Passed: 6 of 6 planted flaws caught and listed first; 0 false alarms on the sound case; model explained 8 of 9 flags. Held-out flaws 2 of 2, but known when the checklist was written, so not a blind test |
| Unit conversion, quote check, placeholder fill-in, checklist verdict | US-8.1–8.4 | Unit tests | Passed |
| Live mode hidden without AWS credentials | US-8.5 | Manual test with credentials removed | Passed |
| Bucket controls in place | US-9.1 | Read back every setting after creation | Passed: public access blocked, ACLs disabled, versioning on, SSE-S3 with SSE-C blocked, HTTPS-only policy, logs to CloudWatch, us-east-2 |
| Planted bad file is quarantined | US-9.2 | 13 files landed, one with prevented > flagged | Passed: 12 months in silver, `month=2027-01/actuals.csv` quarantined with both reasons |
| Gold table equals the app's formula | US-9.3 | Unit test on the exported gold table | Passed: identical savings, adoption, recall and flags; $463,500 delivered (the gold table's own forecast column was computed at the earlier $660,392 estimate; the app recomputes it) |
| Re-run does not double-count | US-9.4 | Run the job twice | Passed: second run loaded 0 rows |

---

## 9. Solution Summary
The Navigator turns a list of AI ideas into a funded, production-ready business case: it **picks the AI that fits** the business need, **proves value** by tuning a real model for dollars instead of accuracy, **checks readiness** so pilots don't stall, and **checks every month after launch** that the promise is kept.

---

## Appendix A: Abbreviations
| Term | Meaning |
|---|---|
| AI BSA | AI business systems analyst |
| ROI | Return on investment |
| AC | Acceptance criteria |
| UC / US | Use case / user story |

## Appendix B: Resources
- AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
