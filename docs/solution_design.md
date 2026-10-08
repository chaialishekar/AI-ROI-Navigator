# AI Use-Case ROI Navigator: Solution Design

*Independent concept project. The document structure follows the public format of Lenovo Validated Design solution guides (Introduction → Business Problem and Business Value → Technical Overview → Architectural Overview → Component Model → Operational Model → Deployment Considerations → Solution Validation → Solution Summary). It is not a Lenovo product and uses no Lenovo data.*

---

## 1. Introduction
The AI Use-Case ROI Navigator is a browser-based decision tool that helps an operations leader:
1. choose which AI use case to start with,
2. prove its value in dollars using a real model, and
3. check whether the organization is ready to move from pilot to production.

**Intended audience:** operations leaders evaluating AI, and the AI business analysts who advise them.

**Hero use case:** predictive maintenance in manufacturing, demonstrated on the public AI4I 2020 dataset.

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
| Faster decisions | Ranks 12 use cases on value vs feasibility in one view | Time from idea list to shortlist |
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
| UC-1 | Prioritize AI use cases | Dana | Leadership asks "where should we start with AI?" | Ranked shortlist with quick wins identified |
| UC-2 | Prove value of predictive maintenance | Dana, with the AI BA | A quick win is selected | Annual $ saved, ROI and payback from a tested model |
| UC-3 | Check production readiness | Dana | Business case is positive | Go / Fix first / Not ready, with top actions |
| UC-4 | Share the business case | Dana | Ready to request funding | One-page summary for the CFO |

**UC-2 detail: Prove value of predictive maintenance**
- **Preconditions:** a trained model and test results are loaded; default cost assumptions are set.
- **Main flow:**
  1. Dana enters her plant's costs (missed failure, false alarm, deployment, running cost).
  2. The tool shows annual net savings, ROI and payback against a "no AI" baseline.
  3. The AI BA moves the alert-threshold slider; failures caught, false alarms and dollars update.
  4. The tool highlights the dollar-optimal threshold next to the accuracy-optimal one.
  5. Dana sets months to deploy and sees first-year value.
- **Alternate flow:** if net savings are negative at every threshold, the tool recommends **do not deploy** and shows which cost drives the loss.
- **Postcondition:** a recommended threshold and a dollar business case are ready for UC-3 and UC-4.

### 3.2a Mapping to the AI lifecycle
Lenovo publicly describes Hybrid AI Advantage as covering the full AI lifecycle: **advisory → implementation → adoption → managed services**, with the goal of moving customers **from pilot to production**. The epics below follow those stages, and every story is tagged with the lifecycle stage and a MoSCoW priority (Must / Should / Could / Won't for v1).

| Lifecycle stage | Customer question | Epics | Built in v1 |
|---|---|---|---|
| **Advisory** | Which AI should we do, and is it worth it? | Epic 1 Prioritize, Epic 2 Prove value | ✅ |
| **Implementation** | Are we ready, and how do we deploy safely? | Epic 3 Readiness, Epic 5 Responsible AI & deployment | ✅ partly |
| **Adoption** | Will people use it, and can we prove it to finance? | Epic 4 Share, Epic 6 Adoption & value tracking | ✅ partly |
| **Managed services** | Is it still delivering value after launch? | Epic 6 Adoption & value tracking | Backlog |

### 3.2 Functional requirements (user stories and acceptance criteria)

#### Epic 1 · Prioritize (UC-1)
| ID | User story | Acceptance criteria |
|---|---|---|
| US-1.1 | As an ops director, I want AI use cases ranked by value and feasibility, so that I know which to start with. | 12 use cases across 3 industries; each scored 1–5 on 7 criteria; each placed in a quadrant; chart numbers match the ranked list. |
| US-1.2 | As an ops director, I want to filter by industry, so that I see only relevant ideas. | Chart and list update immediately. |
| US-1.3 | As an AI BA, I want to adjust weights and scores, so that the ranking fits this customer. | Weight sliders for every criterion; scores accept whole numbers 1–5; ranking updates in under 1 second. |
| US-1.4 | As an ops director, I want a recommended next step per use case, so that I know what to do with it. | Each row shows a next step based on its quadrant. |

#### Epic 2 · Prove value (UC-2)
| ID | User story | Acceptance criteria |
|---|---|---|
| US-2.1 | As an ops director, I want to enter my own costs, so that savings reflect my plant. | Inputs: missed-failure cost, false-alarm cost, deployment cost, yearly running cost, machines in scope. Defaults provided; values must be > 0. |
| US-2.2 | As an ops director, I want annual savings, ROI and payback, so that I can take a business case to my CFO. | Compared against "no AI" (all failures missed); shows net savings, ROI % and payback months; updates on every input change. |
| US-2.3 | As an AI BA, I want to adjust the alert threshold and see the dollar effect, so that I can show accuracy is not the same as value. | Slider; shows recall, precision, caught, missed, false alarms, net $; highlights dollar-optimal vs accuracy-optimal threshold. |
| US-2.4 | As an ops director, I want results in plain language, so that I don't need ML knowledge. | e.g. "Catches 8 of 10 failures. About 1 in 3 alerts is a false alarm." |
| US-2.5 | As an AI BA, I want the model tested on unseen data, so that the numbers are trustworthy. | Train/test split; all reported metrics come from the test set. |
| US-2.6 | As an ops director, I want to compare a custom build with a ready-made solution, so that I see how deployment time changes first-year value. | Input months to deploy; first-year value = annual savings × (12 − months) ÷ 12. |

#### Epic 3 · Readiness (UC-3)
| ID | User story | Acceptance criteria |
|---|---|---|
| US-3.1 | As an ops director, I want a short readiness checklist, so that I know if we're ready to go live. | ≤ 10 questions rated 1–5 across 5 dimensions: data, workflow integration, ROI clarity, user trust, governance and risk. Score per dimension. |
| US-3.2 | As an ops director, I want a verdict and top actions, so that I know what to fix before launch. | **Go:** average ≥ 3.5 and no dimension < 2.5. **Fix first:** any dimension < 2.5. **Not ready:** average < 2.5. Shows 3 weakest dimensions with an action each. |

#### Epic 4 · Share (UC-4)
| ID | User story | Acceptance criteria |
|---|---|---|
| US-4.1 | As an ops director, I want a one-page summary, so that I can share it with my CFO. | Includes use case, annual $ value, ROI, payback, readiness verdict and key assumptions; downloadable. |


#### Epic 5 · Responsible AI & deployment (Implementation)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-5.1 | As an ops director, I want to see why the model flagged a machine, so that technicians trust and act on alerts. | Each alert shows its top 3 drivers (e.g. tool wear, torque) in plain language. | Should (v2) |
| US-5.2 | As an IT lead, I want to choose whether the model runs on-premises, at the edge or in the cloud, so that sensitive plant data stays where our policy requires. | Deployment option is selectable; running cost and data-location note update with the choice. | Could (v2) |
| US-5.3 | As a governance owner, I want an accountable owner and a review trigger for model performance, so that the AI stays safe and accurate after launch. | Readiness check includes owner and monitoring questions; verdict flags gaps. | Must ✅ (covered by US-3.1) |

#### Epic 6 · Adoption & value tracking (Adoption → Managed services)
| ID | User story | Acceptance criteria | Priority |
|---|---|---|---|
| US-6.1 | As an ops director, I want to compare predicted savings with actual results after go-live, so that I can prove the value to finance. | Upload monthly actuals (failures, downtime, false alarms); chart shows predicted vs actual savings. | Should (v2) |
| US-6.2 | As an AI business analyst, I want to track adoption (share of alerts acted on), so that I can spot when users stop trusting the AI. | Adoption rate shown monthly; alert if it drops below 70%. | Should (v2) |
| US-6.3 | As an AI business analyst, I want to be warned when model accuracy drifts, so that we retrain before value drops. | Alert when recall falls 10 points below launch level. | Could (v2) |
| US-6.4 | As an ops director, I want the tool to suggest the next AI use case once the first is live, so that we expand value step by step. | After a "Go" verdict, the summary lists the next 2 quick wins from Epic 1. | Could (v2) |

### 3.3 Non-functional requirements
| Category | Requirement |
|---|---|
| Access | Runs in a web browser; no login |
| Performance | Page loads in under 5 seconds; recalculations in under 1 second |
| Transparency | Every assumption is labeled and editable |
| Trust | Model metrics reported on held-out test data only |
| Branding | Independent project; no company branding |

---

## 4. Architectural Overview
```
User (browser)
   │
   ▼
Streamlit web app ──► Tab 1 Prioritize ──► use_cases.csv + scoring engine
   │                ├► Tab 2 Prove value ──► trained model + test predictions + cost engine
   │                ├► Tab 3 Readiness  ──► checklist + verdict rules
   │                └► Summary          ──► one-page business case (download)
   ▼
Hosted on Streamlit Community Cloud (public link)
```

---

## 5. Component Model
| Component | Responsibility | File |
|---|---|---|
| Use-case catalog | 12 use cases with 7 scores each | `data/use_cases.csv` |
| Scoring engine | Weighted value and feasibility scores, quadrant, next step | `src/scoring.py` |
| Model training | Train and test a failure-prediction model on AI4I 2020 | `src/model.py` |
| Cost engine | Converts caught / missed / false alarms into dollars, ROI and payback | `src/value.py` |
| Readiness engine | Scores dimensions and applies verdict rules | `src/readiness.py` |
| User interface | Tabs, inputs, charts, summary | `app.py` |

---

## 6. Operational Model
How a typical 30-minute session runs:

| Step | Who | Activity | Output |
|---|---|---|---|
| 1. Discover | AI BA with Dana | Agree on business goals; adjust weights | Customer-specific weights |
| 2. Prioritize | Dana | Review the 2×2 grid and ranked list | Shortlist of quick wins |
| 3. Prove value | AI BA with Dana | Enter plant costs; tune the threshold | Dollar business case |
| 4. Check readiness | Dana | Complete the checklist | Verdict and top 3 actions |
| 5. Share | Dana | Download the one-page summary | CFO-ready business case |

---

## 7. Deployment Considerations
- **Hosting:** Streamlit Community Cloud for a free public demo link.
- **Data:** public AI4I 2020 dataset only; no customer data is stored.
- **Customer use:** for real customer data, run the app locally or inside the customer's own environment so data stays on-premises.
- **Dependencies:** listed in `requirements.txt` (Streamlit, pandas, Plotly, scikit-learn).

---

## 8. Solution Validation
Each acceptance criterion is tested before release. Results below; run `pytest` to repeat the automated tests.

| Test | Linked stories | Method | Result |
|---|---|---|---|
| Ranking matches manual calculation for 3 use cases | US-1.1 | Compare app output to hand calculation | ✅ Passed (e.g. predictive maintenance: value 4.75, feasibility 3.80, priority 18.05) |
| Filters and weight changes update instantly | US-1.2, US-1.3 | Manual UI test | ✅ Passed |
| Model metrics come from test data only | US-2.5 | Stratified 70/30 split; all metrics on 3,000 unseen records | ✅ Passed |
| Dollar-optimal threshold differs from accuracy-optimal | US-2.3 | Compare both on the test set (default costs) | ✅ Passed: 0.05 vs 0.90; dollar-optimal saves $29k/year more |
| Verdict rules give the expected result for 3 sample checklists | US-3.2 | Unit tests (`tests/test_core.py`) | ✅ Passed |
| Dollar math matches hand calculation | US-2.2 | Unit test on a 4-record example | ✅ Passed |

---

## 9. Solution Summary
The Navigator turns a list of AI ideas into a funded, production-ready business case: it **prioritizes** use cases on value vs feasibility, **proves value** by tuning a real model for dollars instead of accuracy, and **checks readiness** so pilots don't stall.

---

## Appendix A: Abbreviations
| Term | Meaning |
|---|---|
| AI BA | AI business analyst |
| ROI | Return on investment |
| AC | Acceptance criteria |
| UC / US | Use case / user story |

## Appendix B: Resources
- AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
- Lenovo Validated Design guides (public format reference), Lenovo Press
