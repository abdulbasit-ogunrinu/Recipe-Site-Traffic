# Tasty Bytes · Recipe Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recipe-site-traffic.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64%2B-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Live app → <https://recipe-site-traffic.streamlit.app>**

An interactive dashboard that answers one business question: **which recipes should we feature on the
homepage to drive traffic?** It profiles a recipe catalogue, cleans and explores the data, trains two
classification models against a business recall target, and ships a predictor for pre-screening new
recipes.

This README covers the full project — the business question, the data, the cleaning and modelling
pipeline, the results, and the honest limitations — so you can get the complete picture **without
opening the app**.

---

## Contents

- [Key findings](#key-findings)
- [The business question](#the-business-question)
- [What's in the app](#whats-in-the-app)
- [Metric glossary](#metric-glossary)
- [How it works](#how-it-works)
- [Model card](#model-card)
- [The dataset](#the-dataset)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Deployment](#deployment)
- [Reproducibility](#reproducibility)
- [FAQ](#faq)
- [Limitations and caveats](#limitations-and-caveats)

---

## Key findings

Everything below is reproduced by the live app.

| Finding | Value |
| --- | --- |
| Catalogue after cleaning | **874 recipes** (947 raw − 73 outliers), across 11 categories |
| High-traffic base rate | **60.0%** (524 high / 350 low) — this is the "baseline" every chart compares against |
| Best category | **Vegetable — 98.8%** high-traffic (+38.8 pp vs baseline) |
| Worst category | **Beverages — 5.7%** (−54.3 pp vs baseline) |
| Category spread | **93.1 pp** between best and worst — food type matters more than anything else in the data |
| Best model | **Logistic Regression** — recall **0.848** against the 0.80 target, ROC AUC **0.850** |
| Cost of that recall | 30.0% false-positive rate — roughly 3 in 10 low-traffic recipes get flagged |
| Strongest signals | `protein` (0.169), `calories` (0.152), `carbohydrate` (0.148), `sugar` (0.146) |

**The one-line story:** high-traffic recipes are disproportionately vegetable-, potato- and pork-based
and are more calorie-dense (mean 418 kcal vs 377 kcal for low-traffic), and a simple logistic
regression on nutrition + category is enough to clear the 80% recall bar — a 100-tree random forest
is not.

### High-traffic rate by category

| Category | Recipes | High-traffic rate | vs baseline |
| --- | ---: | ---: | ---: |
| Vegetable | 83 | 98.8% | ▲ +38.8 pp |
| Potato | 82 | 93.9% | ▲ +33.9 pp |
| Pork | 75 | 90.7% | ▲ +30.7 pp |
| Meat | 76 | 75.0% | ▲ +15.0 pp |
| One Dish Meal | 62 | 69.4% | ▲ +9.4 pp |
| Lunch/Snacks | 86 | 65.1% | ▲ +5.1 pp |
| Dessert | 61 | 63.9% | ▲ +3.9 pp |
| Chicken Breast | 89 | 43.8% | ▼ −16.2 pp |
| Chicken | 67 | 37.3% | ▼ −22.7 pp |
| Breakfast | 105 | 31.4% | ▼ −28.6 pp |
| **Beverages** | **88** | **5.7%** | **▼ −54.3 pp** |

## The business question

Tasty Bytes publishes recipes and has one scarce, high-value slot: the **homepage**. Every recipe is
labelled `High` or `Low` traffic. The editorial team wants to promote recipes that will actually
deliver traffic.

The asymmetry drives every design decision in this project:

- **Promoting a low-traffic recipe** wastes the homepage slot — a false positive.
- **Demoting a high-traffic recipe** loses real traffic — a false negative, and the more expensive
  error.

So the business target is stated as **recall ≥ 0.80**: the model must catch at least 80% of the
high-traffic recipes. Precision is reported but deliberately *not* optimised, because the cost of a
miss outweighs the cost of a false flag. The sidebar shows this target as a live progress bar, turning
green when the shipped model clears it.

## What's in the app

Five sections, switched from the sidebar. All charts are interactive Plotly — hover for values, click
legends to filter, drag to zoom — and the ⬇ **Export PDF** button in the top banner prints the current
view (A4 landscape) via the browser print dialog.

| Section | What it gives you |
| --- | --- |
| **📊 Overview Dashboard** | Category / servings / traffic filters, a 7-card KPI ribbon (traffic rate, recall, precision, AUC, catalogue size, categories, median calories) and the core charts: high-traffic rate by category, traffic split, calorie-band mix, category scorecard, rate across calorie bands, and live intelligence alerts. |
| **🔍 Data & EDA** | Cleaning audit (947 raw → 874 clean, 73 outliers dropped, 52 rows imputed), feature distributions, recipes per category, high-vs-low nutrition averages, notched calorie box plots, and mean nutrition by category. |
| **💡 Business Insights** | Baseline vs best vs worst category and the spread between them, popularity by serving size, written key findings and actionable recommendations. |
| **🤖 Model Performance** | Both models side by side across all six metrics against the 0.80 recall target, confusion matrices, ROC curves, Random Forest feature importances, and the full metrics table. |
| **🎯 Predictor** | Enter a recipe's nutrition, servings and category to get a High/Low verdict, a high-traffic probability (≥ 50% → High Traffic), a gauge of the same probability, and a plain-English interpretation with improvement suggestions. |

The sidebar carries a live **BUSINESS TARGET** card showing the 0.80 recall goal, the shipped model's
actual recall, and whether it is on target.

---

## Metric glossary

| Metric | Plain English | Why it matters here |
| --- | --- | --- |
| **Accuracy** | Share of all predictions that were right | Misleading here — a model that always says "high" scores 60% |
| **Precision** | Of the recipes flagged high-traffic, how many really were | The cost of a wasted homepage slot |
| **Recall** | Of the recipes that really were high-traffic, how many we caught | **The business metric.** Target ≥ 0.80 |
| **F1** | Harmonic mean of precision and recall | Balanced summary; useless next to the 0.80 target |
| **ROC AUC** | Ranking quality across all thresholds, 0–1 | 0.5 = coin flip, 1.0 = perfect. Threshold-independent |
| **False Positive Rate** | Share of low-traffic recipes wrongly flagged | Direct measure of wasted slots |
| **pp (percentage points)** | Absolute difference between two percentages | Used for all "vs baseline" deltas — 60% → 75% is +15 pp, not +25% |

## How it works

```
recipe_site_traffic_2212.csv
        │
        ▼
┌───────────────────────────────┐
│ src/data.py :: clean()        │  coerce numerics · parse servings
│  947 rows in                  │  · median-impute by category
└───────────────────────────────┘  · drop |z| ≥ 3 outliers  · binarise target
        │                       →  874 rows out
        ▼
┌───────────────────────────────┐
│ src/model.py :: train_all()   │  ColumnTransformer:
│  stratified 80/20 split,      │   numerics → StandardScaler
│  random_state = 42            │   category → OneHotEncoder
└───────────────────────────────┘
        │              │
        ▼              ▼
 Logistic Regression   Random Forest (100 trees, seed 42)
 recall 0.848          recall 0.724
        │              │
        └──────┬───────┘
               ▼
        app.py  ·  @st.cache_data (trained once per server session)
               ▼
        5 dashboard sections + predictor
```

**Cleaning steps, and what each one is worth**

| Step | Rule | Effect |
| --- | --- | --- |
| ID coercion | `recipe` → int | `001` becomes `1` |
| Numeric coercion | 5 features → float | Non-numeric junk becomes `NaN` |
| Nutrition imputation | Fill gaps with the **per-category median** | 52 rows affected; category-median beats a global median because nutrition profiles differ by food type |
| Servings parsing | Extract the leading integer (`"6"` → `6.0`) | Then median-impute the remainder |
| String normalisation | `category`, `high_traffic` → trimmed lowercase | Makes filtering and mapping predictable |
| Outlier removal | Drop rows where **any** numeric feature has \|z\| ≥ 3 | −73 rows (7.7%); keeps a 914-kcal breakfast from skewing every mean |
| Target binarisation | `high` → 1, everything else → 0 | Produces `high_traffic_label` |

**Features used:** `calories`, `carbohydrate`, `sugar`, `protein`, `servings` (numeric, standardised)
and `category` (one-hot, 11 levels, unknown categories ignored at predict time).

## Model card

**Intended use.** Rank candidate recipes by expected traffic and pre-screen them before publication.
**Out of scope.** Causal claims ("adding protein *causes* traffic"), forecasting future traffic, or
scoring recipes from categories the model has never seen.

**Results** — stratified 80/20 split on 874 rows (175 test rows: 105 high, 70 low), `random_state=42`.

| Model | Accuracy | Precision | **Recall** | F1 | ROC AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Logistic Regression** ✅ | 0.789 | 0.809 | **0.848** | 0.828 | 0.850 | 0.300 |
| Random Forest | 0.703 | 0.768 | 0.724 | 0.745 | 0.821 | 0.329 |

✅ clears the 0.80 recall target.

**Confusion matrices** (rows = actual, columns = predicted)

| | | Pred: Low | Pred: High |
| --- | --- | ---: | ---: |
| **Logistic Regression** | Actual Low | 49 TN | 21 FP |
| | Actual High | **16 FN** | **89 TP** |
| **Random Forest** | Actual Low | 47 TN | 23 FP |
| | Actual High | **29 FN** | 76 TP |

**Why Logistic Regression ships.** It misses 16 high-traffic recipes to the forest's 29, ranks better
(AUC 0.850 vs 0.821), and generalises more predictably — a 100-tree forest on 874 rows across 11
categories mostly memorises noise. The forest is retained as a comparison baseline and for its
feature-importance view.

**The trade, stated plainly.** Recall 0.848 is bought with a 0.300 false-positive rate: about 3 in 10
low-traffic recipes get flagged. For homepage selection that is the correct side to err on, but it
means the output is a **shortlisting signal, not an automatic publish decision**.

## The dataset

`recipe_site_traffic_2212.csv` — 947 rows, one per recipe, shipped with the repo.

| Column | Type | Notes |
| --- | --- | --- |
| `recipe` | id | zero-padded number, e.g. `001` |
| `calories` | float | per serving; contains `NA` |
| `carbohydrate` | float | grams; contains `NA` |
| `sugar` | float | grams; contains `NA` |
| `protein` | float | grams; contains `NA` |
| `category` | string | 11 segments — Breakfast, Beverages, Chicken, Chicken Breast, Dessert, Lunch/Snacks, Meat, One Dish Meal, Pork, Potato, Vegetable |
| `servings` | string → int | stored as text, e.g. `"6"`; values 1, 2, 4, 6 |
| `high_traffic` | string | `High` / `Low` / `NA` — the target |

Raw class balance: 574 `High`, 373 `NA`. After cleaning: 524 high / 350 low across 874 rows.

Derived at runtime: `high_traffic_label` (1/0), plus `z`-scored copies used only inside the outlier
filter.

## Tech stack

- **Streamlit ≥ 1.64** — app framework. The 1.64 floor is required for
  `st.html(unsafe_allow_javascript=True)`, which is how the Export PDF button gets a live click
  handler: Streamlit sanitises HTML with DOMPurify, which strips `onclick` attributes, so the button
  is wired with a delegated `document` listener instead of an inline handler.
- **pandas** / **numpy** — wrangling
- **scikit-learn** — `Pipeline`, `ColumnTransformer`, metrics, feature importances
- **scipy** — z-score outlier detection
- **plotly** — every chart
- Hand-written CSS injected via `st.html` for the dark "control-room" theme, plus
  `.streamlit/config.toml` so the native chrome matches. No UI framework, no external assets except
  the Inter webfont.

## Project structure

```
.
├── app.py                        # Streamlit app: theme, CSS, chart builders, 5 sections
├── src/
│   ├── data.py                   # load_raw / load_cleaned / clean / cleaning_stats
│   └── model.py                  # preprocessing, train_all, predict, metrics
├── recipe_site_traffic_2212.csv  # source data (947 raw rows)
├── notebook.ipynb                # exploratory analysis behind the dashboard
├── requirements.txt              # pinned floors
├── .streamlit/config.toml        # dark theme matching the app palette
├── .python-version               # 3.12, for reproducible Cloud builds
└── README.md
```

`src/` is imported by path insertion (`sys.path`), so the app runs from a clone with no install step
beyond `requirements.txt`.

## Getting started

Requires Python 3.11+ (3.12 recommended).

```bash
git clone https://github.com/abdulbasit-ogunrinu/Recipe-Site-Traffic.git
cd Recipe-Site-Traffic

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. First load trains both models (a few seconds); after that
`@st.cache_data` serves the result, so interactions are instant.

Useful flags while developing:

```bash
streamlit run app.py --server.runOnSave true   # auto-rerun on file save
streamlit run app.py --server.port 8502       # if 8501 is taken
```

## Deployment

The app is live on **Streamlit Community Cloud**:

> **https://recipe-site-traffic.streamlit.app**

It deploys from `main` on GitHub, with `app.py` as the entrypoint and no secrets — the dataset ships
with the repo. Every push to `main` triggers an automatic rebuild, so fixes land on the live URL
within a couple of minutes without any further action.

To deploy your own fork: sign in at <https://share.streamlit.io>, **New app → Deploy to Community
Cloud**, authorise repository access, then pick the fork with `app.py` as the entrypoint. Under
**App settings → General → App URL** you can claim a custom subdomain (6–63 chars); under
**App settings → Sharing →** set *"This app is public and searchable"*. Apps from public repos are
public by default, and public apps are indexed by search engines weekly.

If the app ever shows a build error, open the app's dashboard → **Manage app** → the build/deploy log
names the failing step.

## Reproducibility

- `random_state=42` on the split, the train/test shuffle and the Random Forest
- The test set is a stratified 80/20 split, so class balance is preserved (105 high / 70 low)
- Models train at app start and are cached per server session — every visitor sees identical numbers
- Re-running the app on the same CSV reproduces every figure in this README exactly

To re-derive the numbers without the UI:

```python
import sys; sys.path.insert(0, "src")
import data as d, model as m

df = d.load_cleaned()
info = m.train_all(df)
print(d.cleaning_stats(d.load_raw(), df))
for name, r in info["results"].items():
    print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
```

## FAQ

**Why is the shipped model the simpler one?**
Because the target is recall against a business constraint, and the linear model catches more
high-traffic recipes (0.848 vs 0.724) with better ranking (AUC 0.850 vs 0.821). Complexity didn't pay
for itself on 874 rows.

**Why 80% recall and not accuracy?**
Accuracy is dominated by the 60/40 class balance — always predicting "high" scores 0.60. Recall
measures the thing the business actually cares about: not wasting the homepage slot on a recipe that
would have delivered.

**The predictor says 52% for a recipe I know is weak. Why?**
The 0.5 threshold is a decision rule, not a confidence claim. Read the probability and the category
signal together, and treat anything under ~0.6 as "weak, needs a better angle" rather than a verdict.

**Can I add my own recipe data?**
Replace the CSV, keeping the column names. The cleaning pipeline and both models re-fit on load — no
code changes, no retraining script. Expect the metrics to move; 874 rows is a small sample.

**Why does the first load take a few seconds?**
Both models are trained at startup, including a 100-tree forest. Caching means you only pay it once
per server session.

**Something renders oddly on my machine.**
The theme is pure custom CSS targeting Streamlit's internal `data-testid` attributes, so a very old
or very new Streamlit build may shift a few selectors. `pip install -r requirements.txt` pins the
floors the app was built against.

## Limitations and caveats

- **Missing targets are counted as `low`.** 373 of 947 raw rows have `high_traffic = NA`, and the
  cleaning step maps anything that isn't `high` to `low`. Those recipes are therefore modelled as
  genuine low-traffic recipes. This is the single biggest assumption in the results — **if any of
  those rows are merely unlabelled rather than low, the reported recall is optimistic.** Treat the
  metrics as directional.
- **Small sample.** 874 rows across 11 categories, with cells as small as 61 recipes. Differences of a
  few percentage points between neighbouring categories are not meaningful.
- **Recall is bought with false positives.** A 0.300 FPR means ~3 in 10 low-traffic recipes would be
  flagged. Deliberate, but it makes the output a shortlisting signal rather than an automatic
  decision.
- **Category dominates, and that may be an artefact.** A 93.1 pp spread between Vegetable and
  Beverages is suspiciously large. It likely encodes editorial and merchandising decisions as much as
  reader preference, so "Beverages underperform" may really mean "Beverages weren't promoted" — a
  causal claim the data cannot support.
- **No temporal or editorial features.** No publish date, seasonality, imagery, title style or
  promotion data, all of which plausibly drive real traffic and would very likely be the strongest
  predictors available.
- **A static snapshot.** The data is not refreshed; the app retrains on the shipped CSV every time it
  starts. There is no production pipeline behind it.
- **The 0.5 threshold is unvalidated.** It has not been tuned against real homepage traffic, because
  no outcome data beyond the `High`/`Low` label exists.
