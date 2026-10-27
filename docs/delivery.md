# Delivery log

*Solo project, delivered in time-boxed increments.* Each increment had one goal, shipped something that works end to
end, and was checked by a test or a measured result before the next one started. This is iterative, agile-style
delivery, not Scrum: there was no team, so no stand-ups, product owner or sprint reviews. The "review" at the end of
each increment was a passing test and a working demo. Priorities came from the user stories and MoSCoW tags in
[solution_design.md](solution_design.md); what comes next is in [roadmap.md](roadmap.md).

| # | Dates (commits) | Goal | What shipped | How it was checked | What we learned or changed |
|---|---|---|---|---|---|
| 0 | Oct 8 | A working first version | Use-case scoring, dollar value of a real predictive-maintenance model, readiness check, one-page business case | Ranking and dollar maths match hand calculations (unit tests) | The most accurate alert setting is not the most profitable: tuning for dollars is worth ~$29k a year more |
| 1 | Oct 19 | Show how sure we are | 90% ranges, low / base / high cases, the one assumption that matters most | Range contains the estimate; cases are ordered (unit tests) | Even the pessimistic case pays back; the cost of a breakdown is the number to agree with finance first |
| 2 | Oct 27 | Check the value after launch | Forecast vs actual, adoption and drift alerts, gap split by cause | Gap causes add up exactly to the gap (unit tests on 5 simulated rollouts) | 70% delivered. A forecast is a hypothesis, so it needs a monthly check |
| 3 | Oct 27 | Test where GenAI pays off | Bedrock work-order pilot (30 alerts, two versions); ROI across five cost categories | Graded against the true cause; ROI reproduces a published example | v1 failed (33% right cause). Moving the diagnosis into code fixed it: code does numbers, the model does language |
| 4 | Oct 27 | Measure agents, don't eyeball them | Intake, memo and red-team agents, graded v1 → v3 | 19 cases with known answers, graded in code | Each fix moved a judgement out of the prompt into code. Known limit: the red-team test is not blind |
| 5 | Oct 27 | Real data path for tracking | S3 landing zone → Databricks bronze / silver / gold, quarantine for bad files | Gold table equals the app's formula; a bad file is stopped; a re-run loads 0 rows | Free-tier limits need a sync step; documented the paid-tier route |
| 6 | Oct 27 | Make it understandable in a minute | Overview with the impact up front; every page answer first, analysis on demand | All pages load with no errors; click-through Back / Next; phone and desktop screenshots | First version read as a list of tools; leading with the client and the money fixed it |
| 7 | Oct 27 | Match the right AI to each client | Provider's solution library matched to a client's problem, data, risk appetite and urgency, with reasons | Three example clients get three different answers (unit test) | Reframed the project from "which AI should a company fund" to "which AI fits this client", which is the value engineer's job |

**Practices used throughout:** user stories with acceptance criteria (36+), MoSCoW priorities, value × feasibility
scoring (also used to prioritise the roadmap), a fishbone of why value falls short and a RACI for who owns what after
launch (both in [solution_design.md](solution_design.md) §6), and the red-team agent as an automated pre-mortem.
