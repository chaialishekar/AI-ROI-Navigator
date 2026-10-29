# The ROI Compass: an AI business case, end to end

> Which AI should a business buy first, and will it pay off?

[![Live site](https://img.shields.io/badge/Live_site-GitHub_Pages-0f6d74)](https://chaialishekar.github.io/AI-ROI-Navigator/)
![Python](https://img.shields.io/badge/Python-3.12-111111)
![Data](https://img.shields.io/badge/Data-UCI_AI4I_2020-6b6b6b)
![GenAI](https://img.shields.io/badge/GenAI-Amazon_Bedrock-6b6b6b)

**[Open the website](https://chaialishekar.github.io/AI-ROI-Navigator/)** · **[Try the calculator](https://chaialishekar.github.io/AI-ROI-Navigator/phases/business-case.html#details)** · **[2-page decision brief](docs/decision_brief_sample.pdf)** · **[PRD](docs/prd.md)** · **[Architecture](ARCHITECTURE.md)**

Most companies buying AI can't say whether it paid off. In [IBM's 2025 CEO Study](https://newsroom.ibm.com/2025-05-06-ibm-study-ceos-double-down-on-ai-while-navigating-enterprise-hurdles), only 25% of AI projects delivered the return expected. So I took one factory and worked the question through from start to finish: which AI tool to buy first, whether it beats a simple rule or doing nothing, and whether the savings actually showed up a year later.

![Home page](docs/site_home.png)

## The answer in three numbers

| | A year |
|---|---|
| What the model predicted, tested on 3,000 machine records | **$658k** |
| What finance planned on, from published studies | **$290k** |
| What was delivered in the first year (simulated) | **$464k** |

The plan was beaten, but only 70% of the model's estimate showed up. The Measure phase explains the gap: fewer breakdowns than assumed, crews ignoring some alerts at first, and the model getting weaker over time.

## Seven phases, one answer each

| Phase | Question | Answer |
|---|---|---|
| 01 Discover | What's going wrong, and what does it cost? | $3.7M a year lost to downtime, scrap and shortages |
| 02 Evaluate | Which AI should they buy first? | Predictive maintenance (warning before a machine breaks). The runner-up rests on a vendor's own claim |
| 03 Business case | Will it pay for itself? | Plan on $290k a year; still $162k in the worst case |
| 04 Readiness | Can the team use it? | Not yet: crews need warnings in their work system, and need to trust them |
| 05 Decide | What is finance signing? | Not "buy it": pilot on one production line, after the gaps are closed |
| 06 Deliver | What gets built? | Requirements, 13 acceptance tests, a working model and data pipeline |
| 07 Measure | Did the savings show up? | $464k against the $290k plan, with the shortfall explained |

One honest twist: the maintenance manual's simple rules save slightly more on paper ($721k). But they raise about 85 false alarms a year against the AI's 13, and crews stop answering alarms that are usually wrong. The website shows where that trade-off flips.

## What's real and what's illustrative

- **Real:** the machine data ([AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), 10,000 records), the model trained on it, the S3 → Databricks data pipeline, and two generative AI tests on Amazon Bedrock.
- **Published, with sources:** how much each AI tool improved results elsewhere, labelled by how strong the evidence is.
- **Illustrative:** the factory's costs, the vendor's prices and the team ratings (all editable on the website), and the year after launch (simulated).
- **A caution:** the dataset is synthetic, so breakdowns are easier to predict than in a real plant. That's why the budget is set on the published $290k, not the model's $658k.

## On the website

| Seven phases as a staircase | Every option priced for one plant |
|---|---|
| ![Staircase](docs/site_staircase.png) | ![Evaluate options](docs/site_evaluate.png) |

**A calculator you can change.** Try a what-if and watch a CFO, COO and CTO react.

![Calculator](docs/site_calculator.png)

| Team readiness, rated live | A year after go-live |
|---|---|
| ![Readiness](docs/site_readiness.png) | ![Measure](docs/site_measure.png) |

## Two side experiments with generative AI

- **Drafting repair orders from warnings** (30 cases): when the AI diagnosed the fault itself it was right 10 times in 24; when code diagnosed and the AI only wrote, 24 in 24. A person still reviews every draft.
- **Reviewing a business case for mistakes** (19 cases): after the same split (code checks, the AI explains), it caught all 6 planted errors. I wrote the test cases, so it is not a blind test.

![GenAI results](docs/site_genai.png)

## Documents

| Document | What's in it |
|---|---|
| [Product requirements (PRD)](docs/prd.md) | The problem, goals, users, user stories, success measures, risks |
| [Decision brief (PDF)](docs/decision_brief_sample.pdf) | The two pages finance signs |
| [Requirements pack](docs/requirements.md) | Process maps, how a warning becomes a work order, 13 acceptance tests |
| [Solution design](docs/solution_design.md) | Stakeholders, 36+ user stories, who owns what after launch |
| [Roadmap](docs/roadmap.md) · [Delivery log](docs/delivery.md) | What comes next, and how the work was delivered |
| [Architecture](ARCHITECTURE.md) | How the numbers are calculated and how the pieces fit together |

## Run it locally

```bash
pip install -r requirements.txt
python -m http.server -d site 8000   # the website, at localhost:8000
streamlit run app.py                 # the analyst workbench, including live Bedrock runs
pytest                               # 23 tests
```

After changing anything in `src/` or `data/`, refresh the website's numbers with `python site/export_data.py`.

## Credits

Data: Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository, CC BY 4.0.

Built by Chaitali Shekar · MS Engineering Management, NC State University. An independent project: the analysis, the model, the data pipeline, the AI tests and the website.
