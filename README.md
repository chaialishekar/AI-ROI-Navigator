# AI Use-Case ROI Navigator

> Most teams judge AI models by accuracy. This tool judges them by dollars.

Most AI projects stall between pilot and production: companies pick projects on hype, judge models by accuracy instead of business value, and launch before they're ready. The ROI Navigator walks an operations leader through four steps, then checks the promise after launch:

| Tab | Question it answers |
|---|---|
| **1 · Prioritize** | Which AI use case should we do first? Scores 12 use cases on value vs feasibility. |
| **2 · Prove value** | What is it worth, and how sure are we? Tunes a real predictive-maintenance model for **dollars saved, not accuracy**, with uncertainty ranges and low / base / high cases. |
| **3 · Readiness** | Are we ready to go live? 10-question scorecard with a Go / Fix first / Not ready verdict. |
| **4 · Business case** | One-page summary to share with finance (downloadable). |
| **5 · Track value** | Is it delivering what we promised? Compares actual savings with the forecast month by month, warns when the crew stops acting on alerts or the model drifts, and explains the gap. |

*Independent concept project. Use-case scores and plant costs are illustrative assumptions and can be edited in the app.*

## Key result (default assumptions)
On 3,000 machine records the model never saw during training, a gradient-boosting model:
- catches **~86% of failures** with **~80% of alerts being real**
- yields **~$660K annual net savings** for a plant with 40 failures a year, with a **~4.7-month payback**
- **The result holds up under pessimistic assumptions.** Model uncertainty puts annual savings at **$611K–$704K** (90% range). In the low case, with every cost assumption at its unfavorable end, the project still saves **~$164K a year** and pays back in **~20 months**. The biggest driver is the cost of an unplanned failure, so that's the number to agree with finance first.
- **Forecasts need checking after launch.** In a simulated 12-month rollout the plant banks **$464K of the $660K forecast (70%)**. The Track value tab splits the $197K gap into fewer failures than assumed (−$91K), the crew ignoring correct alerts in the first months (−$63K) and model drift (−$46K), and raises a retrain alert in month 11.
- **The most accurate setting is not the most profitable.** Tuning the alert threshold for dollars instead of accuracy saves **~$29K more per year**, because a missed failure ($25K) costs far more than a false alarm ($500).

| Prioritize | Prove value |
|---|---|
| ![](docs/tab1_prioritize.png) | ![](docs/tab2_prove_value.png) |
| **Readiness** | **Business case** |
| ![](docs/tab3_readiness.png) | ![](docs/tab4_business_case.png) |

**Track value** (simulated 12-month rollout)

![](docs/tab5_track_value.png)

## How it works
- **Prioritization:** weighted 1–5 scores. Value = financial impact, strategic fit, scale. Feasibility = data readiness, ease of build, low risk, speed to value. Weights are adjustable.
- **Model:** AI4I 2020 dataset (10,000 records, 339 failures). Features: temperatures, speed, torque, tool wear, plus engineered power, temperature difference and wear × torque. Failure-mode columns are excluded to avoid leakage. Logistic regression (explainable baseline) vs gradient boosting; stratified 70/30 split.
- **Dollars:** test-set catch, miss and false-alarm rates are scaled to the plant's failures per year. Savings = cost of failures without AI − (planned repairs + missed failures + false alarms + running cost).
- **Uncertainty:** *model uncertainty* comes from a bootstrap of the test set (1,000 resamples, 90% range). *Assumption uncertainty* comes from a tornado chart that moves each cost input between an editable low and high value one at a time, plus low / base / high cases that set every input to its unfavorable or favorable end.
- **Value tracking:** monthly counts (alerts, failures flagged, prevented, missed, false alarms) are turned into dollars with the same formula as the forecast. Adoption = share of alerts acted on (alert below 70%). Drift = 3-month rolling recall 10+ points below launch. The gap is split by swapping each forecast assumption (failure volume, recall, share of flagged failures acted on, false alarms) for its actual value one at a time, so the parts add up exactly. The bundled 12-month rollout is simulated, not real plant data.
- **Readiness:** 5 dimensions (data, workflow fit, ROI clarity, user trust, governance). Go = average ≥ 3.5 and no dimension below 2.5.

Business analysis docs (personas, use cases, 22 user stories with acceptance criteria, validation) are in [`docs/solution_design.md`](docs/solution_design.md).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest            # optional: run the tests
```

## Data
Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. Licensed CC BY 4.0.

## Author
Chaitali Shekar · MS Engineering Management, NC State University
