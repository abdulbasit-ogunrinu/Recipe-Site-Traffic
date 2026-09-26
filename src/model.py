import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["calories", "carbohydrate", "sugar", "protein", "servings"]
CATEGORICAL_FEATURES = ["category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_NAMES = ["Logistic Regression", "Random Forest"]

CONFUSION_LABELS = ["Low Traffic", "High Traffic"]


def _build_preprocessor():
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def _build_pipelines():
    preprocessor = _build_preprocessor()
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        ),
    }


def train_all(clean_df, random_state=42, test_size=0.2):
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["high_traffic_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    roc_data = {}
    feature_importance = None

    for name, pipe in _build_pipelines().items():
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics["False Positive Rate"] = fp / (fp + tn)
        metrics["confusion_matrix"] = cm
        metrics["model"] = pipe

        results[name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics["ROC AUC"]}

        if name == "Random Forest":
            rf = pipe.named_steps["classifier"]
            ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
            feature_names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
            imp = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
            feature_importance = imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "results": results,
        "roc_data": roc_data,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict(info, model_name, features):
    pipe = info["results"][model_name]["model"]
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    proba = float(pipe.predict_proba(row)[:, 1][0])
    label = 1 if proba >= 0.5 else 0
    return label, proba