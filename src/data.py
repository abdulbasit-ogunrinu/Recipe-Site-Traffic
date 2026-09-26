from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC_FEATURES = ["calories", "carbohydrate", "sugar", "protein", "servings"]
NUTRITION_FEATURES = ["calories", "carbohydrate", "sugar", "protein"]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "recipe_site_traffic_2212.csv"


def load_raw(path=DEFAULT_DATA_PATH):
    return pd.read_csv(path)


def load_cleaned(path=DEFAULT_DATA_PATH):
    return clean(load_raw(path))


def clean(raw_df):
    df = raw_df.copy()

    df["recipe"] = pd.to_numeric(df["recipe"], errors="coerce").astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in NUTRITION_FEATURES:
        df[col] = df.groupby("category")[col].transform(lambda x: x.fillna(x.median()))

    for col in ["category", "high_traffic"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df["servings"] = (
        df["servings"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["servings"] = df.groupby("category")["servings"].transform(
        lambda x: x.fillna(x.median())
    )

    z = np.abs(stats.zscore(df[NUMERIC_FEATURES]))
    df = df[(z < 3).all(axis=1)]

    df["high_traffic"] = df["high_traffic"].apply(
        lambda x: "high" if str(x).lower() == "high" else "low"
    )
    df["high_traffic_label"] = df["high_traffic"].map({"low": 0, "high": 1})

    return df.reset_index(drop=True)


def cleaning_stats(raw_df, clean_df):
    nutrition_missing = int(raw_df[NUTRITION_FEATURES].isna().any(axis=1).sum())
    removed_outliers = len(raw_df) - len(clean_df)
    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "removed_outliers": removed_outliers,
        "nutrition_missing": nutrition_missing,
    }