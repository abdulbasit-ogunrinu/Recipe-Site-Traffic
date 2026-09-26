# Tasty Bytes · Recipe Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recipe-site-traffic.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64%2B-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Live app → <https://recipe-site-traffic.streamlit.app>**

An interactive dashboard that answers one business question: **which recipes should we feature on the
homepage to drive traffic?** It profiles a recipe catalogue, cleans and explores the data, trains two
classification models against a business recall target, and ships a predictor for pre-screening new
recipes.

This README is written so you can get the full picture **without opening the app** — every number,
chart and control is explained below. If you'd rather explore it yourself, the
[dashboard tour](#dashboard-tour) tells you exactly what you're looking at, section by section.

---

## Contents

- [Key findings](#key-findings)
- [The business question](#the-business-question)
- [Dashboard tour](#dashboard-tour)
  - [Sidebar and top banner](#sidebar-and-top-banner)
  - [1 · Overview Dashboard](#1--overview-dashboard)
  - [2 · Data & EDA](#2--data--eda)
  - [3 · Business Insights](#3--business-insights)
  - [4 · Model Performance](#4--model-performance)
  - [5 · Predictor](#5--predictor)
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

## Dashboard tour

The app is a single page with five sections, switched from the sidebar. All charts are Plotly
(interactive — hover for values, click legends to filter, drag to zoom). The whole page re-scopes
whenever a control changes.

### Sidebar and top banner

**Sidebar (top to bottom)**

- **Brand block** — the Tasty Bytes logo card.
- **NAVIGATE** — radio buttons for the five sections.
- **BUSINESS TARGET card** — the 0.80 recall target with a progress bar showing the shipped model's
  actual recall and a status line (`✓ Status: On Target` / `⚠ Status: Below Target`). This is the
  single number the project is judged on.
- **Footer** — source dataset name and the models in play.

Closing the sidebar leaves a floating **orange ⏵ button** in the top-left of the header to reopen it.

**Top banner** — always visible above the content: title, dataset pills (recipe count, category
count, recall target) and a **model status badge**. On the right, the **⬇ Export PDF** button calls
`window.print()` with a dedicated print stylesheet (A4 landscape, sidebar and chrome hidden, dark
theme preserved via `print-color-adjust: exact`, cards and tables kept off page breaks). Choose
*Save as PDF* as the print destination.

### 1 · Overview Dashboard

The executive view. Everything responds to three filters.

**Filters (top row)**

| Filter | Options | Effect |
| --- | --- | --- |
| Category · Segment | All + 11 categories | Restricts to one food segment |
| Servings · Period | All, 2, 4, 6 | Restricts to a serving-size group |
| Traffic · Region | All, High, Low | Restricts to one outcome class |

If a combination returns nothing, the app warns you and falls back to the full catalogue rather than
rendering empty panels.

**KPI ribbon — 7 ring cards.** Each ring is filled to the share it represents; the sub-line gives
context.

| Card | What it measures | How to read it |
| --- | --- | --- |
| High-Traffic Rate | Share of the current view labelled high | The headline outcome. Its delta is percentage points vs the 60.0% catalogue baseline |
| Model Recall | Shipped model's recall | Green if ≥ 80% target, amber if not |
| Precision | Share of flags that were correct | The cost side of the recall trade-off |
| ROC AUC | Discriminative power, 0–1 | How well the model ranks high above low overall; 0.850 is strong |
| Total Recipes | Rows in the current view | Sub-line shows share of the 874-recipe catalogue |
| Categories | Distinct categories in view | Sub-line shows share of 11 |
| Median Calories | Median kcal per recipe | Sub-line notes it is post-outlier-removal |

**Charts**

1. **High-Traffic Rate by Category** — bars per category for the current view against the catalogue
   baseline. Orange bars beat the baseline; cyan bars fall short. This is the chart that answers
   "where should we spend homepage slots".
2. **Traffic Split** — donut of high vs low in the view, with two progress bars underneath giving the
   raw counts.
3. **Calorie Bands by Category** — 100% stacked bars splitting each category into light / moderate /
   rich calorie bands, so you can see *why* a category over- or under-performs.
4. **Category Scorecard** — a sortable-feeling table: category, recipe count, high-traffic rate, ▲/▼
   movement in percentage points vs baseline, and share of the view.
5. **Traffic Rate across Calorie Bands** — line over sequential calorie bands with the peak band
   marked by an orange diamond. Shows whether there's a sweet spot in calories.
6. **High-Traffic Share by Category** — top-5 progress bars of where the high-traffic recipes in the
   view actually come from, followed by three live **Intelligence Alerts**: the view's leading
   category, the one to deprioritise, and the model's recall status.

**Bottom strip** — five one-line answers: current run rate, top category, fastest growth vs
baseline, model recall status, and a watchlist category.

### 2 · Data & EDA

Provenance and exploration — the "can I trust this?" page.

**Cleaning audit cards:** Raw Rows (947) · Cleaned Rows (874) · Outliers Removed (73, 7.7%, any
feature with |z| ≥ 3) · Missing Imputed (52 rows, category-median fill on nutrition).

**Charts**

- **Feature Distributions** — histogram of a selectable numeric feature (calories / carbohydrate /
  sugar / protein). Look for skew and heavy tails; the calories distribution runs 0.14 → 1,724 kcal
  with a median of 291.
- **Recipes per Category** — counts per category, largest first (Breakfast 105 → One Dish Meal 61).
- **Nutrition Averages: High vs Low Traffic** — grouped bars of mean nutrient values split by class.
  High-traffic recipes average 418 kcal / 32.5 g carbs / 6.7 g sugar / 20.3 g protein against
  377 / 28.3 / 8.0 / 20.4 for low — **more calories, less sugar**, and effectively identical protein.
- **Calorie Distribution by Traffic Type** — notched box plots. Overlapping notches mean the median
  difference is not statistically significant at conventional levels; the visible rightward shift of
  the high-traffic box is the calorie effect.
- **Mean Nutrition Values by Category** — grouped bars comparing all four nutrients across every
  category.

### 3 · Business Insights

The "so what" page, with no model machinery in the way.

**Cards:** Baseline Rate (60.0%) · Best Category (Vegetable) · Weakest Category (Beverages) ·
Category Spread (93.1 pp, best minus worst — the headline measure of how much category choice matters).

**Charts**

- **High-Traffic Rate by Category** — orange bars exceed the dotted baseline line; this is the
  same ranking as the table above, sorted.
- **Popularity Rate by Serving Size** — 1, 2, 4 and 6 servings against the baseline. 2–4 serving
  recipes skew higher, which is the basis for the "small groups" recommendation.

**Key Findings** — homepage prioritisation (lead with Vegetable), review Beverages, the 2–4 serving
sweet spot, and a model-confidence statement.

**Actionable Decisions** — pre-screen new recipes with the predictor, monitor recall and precision
monthly and retrain as data accumulates, diversify the homepage across high-performing categories,
and A/B test presentation for weak categories before writing them off.

### 4 · Model Performance

Everything needed to judge whether the model is fit to deploy.

**Cards:** LR Recall · LR Precision · LR ROC AUC · RF Recall · RF ROC AUC.

**Performance Comparison** — grouped bars of all six metrics for both models, with a dotted line at
80% marking the recall target. The shipped model is the one that clears it.

**Confusion Matrix tab** — pick a model, then read a 2×2 heatmap (rows = actual, columns = predicted).
For Logistic Regression on 175 held-out rows: 89 true positives, 49 true negatives, 21 false
positives, 16 false negatives. The 16 in the bottom-left cell are the recipes the model would have
wrongly passed over — the errors the 0.80 target exists to minimise.

**ROC Curves tab** — true-positive rate against false-positive rate at every threshold, with the
diagonal drawn in. The further the curve hugs the top-left, the better the ranking; AUC 0.850 for LR
vs 0.821 for RF.

**Feature Importance** — top 10 Random Forest predictors, teal → orange. Nutrition features
dominate, and `category_beverages` / `category_vegetable` show up as the strongest categorical
effects.

**Metrics Table** — the full scorecard for both models, heat-mapped per row so the better value in
each pair lights up.

### 5 · Predictor

Pre-screen a recipe before you spend a homepage slot on it.

**Inputs** (a form, so nothing recomputes until you press **▶ Run Prediction**): Calories (0–3,000),
Carbohydrate (0–500 g), Sugar (0–300 g), Protein (0–200 g), Servings (1 / 2 / 4 / 6), Category (all
11). Defaults are the catalogue medians.

**Outputs**

- **Prediction card** — `High Traffic` or `Low Traffic`, with the model's confidence in that call.
- **High-Traffic Probability** — the raw probability, with the decision rule stated: **≥ 50% → High
  Traffic**.
- **Gauge** — the same probability as an arc, with an orange threshold line at the 50% decision
  boundary.
- **Interpretation** — plain-English operational guidance: *Recommend for Homepage* with the
  probability restated, the *Category Signal* (that category's own high-traffic rate, so you can see
  whether the model is leaning on category or nutrition), and either a *Headroom* note or an
  *Improvement Suggestion* telling you which direction to move the nutrition numbers.

**How to use it as a sensitivity tool.** Leave everything at the defaults and press the button —
that's the median recipe. Then move one variable at a time and watch the probability. Category is
usually the strongest lever (try Beverages vs Vegetable at identical nutrition). Because the model is
linear in the nutrition features, its response to those is smooth and monotonic; only the category
dummy produces jumps.

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
