# Tasty Bytes · Recipe Intelligence

An interactive Streamlit dashboard that answers one business question: **which recipes should we
feature on the homepage to drive traffic?** It profiles the recipe catalogue, cleans and explores the
data, and trains classification models that score a recipe's likelihood of becoming a high-traffic
page.

<!-- Deploy badge: after going live on Streamlit Community Cloud, replace YOUR_USERNAME below
     and confirm the URL matches https://<user>-<repo>.streamlit.app/ -->
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_USERNAME-recipe-site-traffic.streamlit.app)

---

## Table of contents

- [Overview](#overview)
- [Live demo](#live-demo)
- [Features](#features)
- [Model performance](#model-performance)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [The dataset](#the-dataset)
- [Caveats and limitations](#caveats-and-limitations)

---

## Overview

**Tasty Bytes** is a recipe publisher deciding which recipes earn the scarce homepage slot. Traffic is
labelled `High` or `Low` per recipe, and the business target is a model that **catches at least 80% of
high-traffic recipes (recall ≥ 0.80)** — a miss is a wasted homepage slot, so recall is prioritised
over precision.

The app is a single-page dashboard with five sections, all driven by a live logistic-regression model
that is trained on load and cached with `@st.cache_data`.

## Live demo

**[→ Open the deployed app](https://YOUR_USERNAME-recipe-site-traffic.streamlit.app)**

> Replace the placeholder URL above with your deployed Streamlit Community Cloud address.

To run it locally instead:

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

## Features

| Section | What it does |
| --- | --- |
| **📊 Overview Dashboard** | Category / servings / traffic filters that re-scope the whole page, a 7-card KPI ribbon (traffic rate, recall, precision, AUC, catalogue size, categories, median calories), plus traffic-vs-target, market share, calorie-band mix, category scorecard and intelligence alerts. |
| **🔍 Data & EDA** | Cleaning audit (raw rows, cleaned rows, outliers dropped, values imputed) and the full exploratory pass: feature distributions, recipes per category, high-vs-low nutrition averages, calorie distributions and mean nutrition by category. |
| **💡 Business Insights** | High-traffic rate by category, popularity by serving size, a ranked category scorecard, and a written set of findings and recommendations. |
| **🤖 Model Performance** | Side-by-side comparison of Logistic Regression and Random Forest across accuracy, precision, recall, F1, ROC AUC and false-positive rate, with confusion matrices, ROC curves, Random Forest feature importances and a full metrics table. |
| **🎯 Predictor** | Interactive form — pick a category and serving size, tune calories / carbs / sugar / protein — and get a predicted label, a high-traffic probability ring, a gauge of the model's confidence and a plain-English interpretation with improvement suggestions. |

**Export PDF** — the ⬇ button in the top-right banner opens the browser print dialog with a print
stylesheet applied (landscape A4, sidebar and chrome hidden, dark theme preserved, cards and tables
kept off page breaks). Use *Save as PDF* as the destination.

## Model performance

Trained on a stratified 80/20 split, `random_state=42`, on 874 cleaned rows (524 high / 350 low):

| Model | Accuracy | Precision | **Recall** | F1 | ROC AUC | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** ✅ | 0.789 | 0.809 | **0.848** | 0.828 | 0.850 | 0.300 |
| Random Forest | 0.703 | 0.768 | 0.724 | 0.745 | 0.821 | 0.329 |

✅ = meets the 0.80 recall target. **Logistic Regression is the model the app ships**, because recall
against the business target matters more than raw accuracy, and the linear model generalises better on
a dataset this small. Both models clear the target-adjacent bar; the Random Forest is retained for
comparison and for its feature-importance view.

Top Random Forest features: `protein` (0.169), `calories` (0.152), `carbohydrate` (0.148),
`sugar` (0.146), `category_beverages` (0.079).

## How it works

1. **Load** — `src/data.py:load_raw` reads the CSV; `recipe` is coerced to a numeric id.
2. **Clean** — `src/data.py:clean`:
   - coerces the five numeric features, imputing nutrition gaps with the **per-category median**;
   - parses `servings` out of its string form (`"6"` → `6`) and imputes the remainder by median;
   - normalises `category` and `high_traffic` to lowercase strings;
   - drops **z-score outliers** (`|z| ≥ 3` on any numeric feature) — 73 of 947 rows;
   - derives `high_traffic_label` (high → 1, low → 0).
3. **Model** — `src/model.py:train_all` builds two scikit-learn pipelines sharing a
   `ColumnTransformer` (standard-scaled numerics + one-hot category), fits both, and returns
   metrics, ROC curves, the confusion matrices and feature importances.
4. **Serve** — `app.py` renders the dashboard; `@st.cache_data` means the models are trained once
   per server session, not per interaction.

## Tech stack

- **Streamlit** ≥ 1.64 — app framework (1.64+ required for `st.html(unsafe_allow_javascript=True)`,
  which powers the Export PDF button)
- **pandas** / **numpy** — data wrangling
- **scikit-learn** — pipelines, preprocessing, metrics, RandomForest feature importances
- **scipy** — z-score outlier detection
- **plotly** — all charts
- Hand-written CSS injected through `st.html` for the dark "control-room" theme; no external UI kit

## Project structure

```
.
├── app.py                        # Streamlit app: layout, CSS, all five sections
├── src/
│   ├── data.py                   # loading + cleaning pipeline
│   └── model.py                  # preprocessing, training, metrics, prediction
├── recipe_site_traffic_2212.csv  # source dataset (947 raw rows)
├── notebook.ipynb                # exploratory analysis behind the dashboard
├── requirements.txt
├── .streamlit/config.toml        # dark theme, matches the app's palette
└── README.md
```

## Getting started

Requires Python 3.11+.

```bash
git clone https://github.com/YOUR_USERNAME/recipe-site-traffic.git
cd recipe-site-traffic

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens on <http://localhost:8501>. It trains both models on first load (a few seconds) and
caches the result, so subsequent reloads are instant.

## Deploying to Streamlit Community Cloud

The app is deployment-ready — `app.py` sits at the repo root and `requirements.txt` lists every
dependency.

1. **Push the repo to GitHub** (make it public — the free Streamlit Community Cloud tier requires it).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app → Deploy to Community Cloud**, then authorise Streamlit to access your repositories.
4. Select `YOUR_USERNAME/recipe-site-traffic` and branch `main`. Leave the entrypoint as
   `app.py` and the deploy path empty.
5. Click **Deploy**. The app builds in ~2 minutes at
   `https://YOUR_USERNAME-recipe-site-traffic.streamlit.app`.
6. Copy that URL into the *Live demo* link and the badge at the top of this README.

There are no secrets, API keys or external services to configure — the dataset ships with the repo.

## The dataset

`recipe_site_traffic_2212.csv` — 947 rows, one per recipe:

| Column | Type | Notes |
| --- | --- | --- |
| `recipe` | id | zero-padded recipe number |
| `calories`, `carbohydrate`, `sugar`, `protein` | float | per serving; contain `NA` gaps |
| `category` | string | 11 segments: Breakfast, Beverages, Chicken, Chicken Breast, Dessert, Lunch/Snacks, Meat, One Dish Meal, Pork, Potato, Vegetable |
| `servings` | int | stored as a string, e.g. `"6"` |
| `high_traffic` | string | `High` / `Low` / `NA` — the classification target |

Composition: 574 `High`, 373 `NA` in the raw file; 11 categories, largest is Breakfast (106) and
smallest One Dish Meal (71).

## Caveats and limitations

- **Missing targets are counted as `low`.** 373 of the 947 raw rows have `high_traffic = NA`; the
  cleaning step maps anything that is not `high` to `low`. Those recipes are therefore modelled as
  genuine low-traffic recipes, which is the single biggest assumption in the results — the reported
  recall is optimistic if any of those rows are actually unlabelled rather than low.
- **Small sample.** 874 cleaned rows across 11 categories; treat the metrics as directional, and
  prefer the linear model — a 100-tree forest on this much data mostly memorises noise.
- **Recall is bought with false positives.** A 0.30 FPR means roughly 3 in 10 low-traffic recipes
  would be flagged; that is the deliberate trade for clearing the 0.80 recall target.
- **No temporal or editorial features.** The model sees nutrition and category only — no publish
  date, seasonality, image quality or promotion, all of which plausibly drive real traffic.
- **Static snapshot.** The data is not refreshed; re-running the app retrains on the shipped CSV.
