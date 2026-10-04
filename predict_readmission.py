"""
Diabetes hospital readmission prediction (within 30 days).

Dataset: UCI Diabetes 130-US Hospitals (1999-2008), 101,766 encounters.
Target: readmitted == "<30"  -> 1, otherwise 0.

Models compared: Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, and K-Nearest Neighbors (all with balanced handling of
the minority class where the model supports it, because only ~11% are
readmitted <30d).

To keep run time manageable we train on a stratified random sample of
30,000 encounters (stated in README). Remove SAMPLE_N to use all data.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://raw.githubusercontent.com/ibrahimalfawaz/"
    "Cambridge-Module-5/main/diabetic_data.csv"
)
SAMPLE_N = 30_000
RANDOM_STATE = 42
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL = [
    "age", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "A1Cresult", "insulin", "change", "diabetesMed",
    "metformin", "glyburide", "glipizide",
]
FEATURES = NUMERIC + CATEGORICAL


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, low_memory=False)
    df = df.replace("?", np.nan)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df


def chart_readmission_by_age(df: pd.DataFrame) -> None:
    order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    rate = df.groupby("age", observed=True)["target"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rate.plot(kind="bar", ax=ax, color="#2a7f62")
    ax.set_ylabel("Readmitted <30 days (rate)")
    ax.set_xlabel("Age group")
    ax.set_title("30-day readmission rate by age group (full dataset)")
    plt.tight_layout()
    fig.savefig(OUT / "readmission_rate_by_age.png", dpi=150)
    plt.close(fig)


def chart_readmission_by_prior_inpatient(df: pd.DataFrame) -> None:
    binned = df.copy()
    binned["prior_inpatient_bin"] = pd.cut(
        binned["number_inpatient"], bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"])
    rate = binned.groupby("prior_inpatient_bin", observed=True)["target"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rate.plot(kind="bar", ax=ax, color="#305496")
    ax.set_ylabel("Readmitted <30 days (rate)")
    ax.set_xlabel("Prior inpatient visits")
    ax.set_title("30-day readmission rate by prior inpatient visits (full dataset)")
    plt.tight_layout()
    fig.savefig(OUT / "readmission_rate_by_prior_inpatient.png", dpi=150)
    plt.close(fig)


def evaluate(name, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def chart_feature_importance(model, feature_names, model_name) -> None:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print(f"Skipping feature-importance chart: {model_name} has neither "
              "feature_importances_ nor coefficients.")
        return
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([feature_names[i] for i in idx][::-1],
            importances[idx][::-1], color="#7f4f24")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 features — {model_name}")
    plt.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def chart_confusion_matrix(cm, model_name) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No", "Pred: Yes"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No", "Actual: Yes"])
    ax.set_title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df_full = load_data()
    full_rows = int(len(df_full))
    full_pos_rate = round(float(df_full["target"].mean()), 4)
    print(f"Full dataset: {full_rows:,} encounters, "
          f"positive rate = {full_pos_rate:.3f}")
    df = df_full

    chart_readmission_by_age(df)
    chart_readmission_by_prior_inpatient(df)

    if SAMPLE_N and len(df) > SAMPLE_N:
        df_model, _ = train_test_split(
            df, train_size=SAMPLE_N, stratify=df["target"],
            random_state=RANDOM_STATE)
        print(f"Training sample: {len(df_model):,} encounters (stratified)")
    else:
        df_model = df

    X = df_model[FEATURES]
    y = df_model["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE),
        # KNN is distance-based, so it relies on the StandardScaler in the
        # preprocessing pipeline. It has no class_weight, so the features
        # being scaled is what gives it a fair comparison here.
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }

    results = []
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        res = evaluate(name, pipe, X_test, y_test)
        results.append(res)
        print(res)

    # Screening choice: for follow-up triage, recall (catching true
    # readmissions) matters most. The screening model is the one with the
    # highest recall; ties/close calls are broken by ROC-AUC. This guards
    # against a high-accuracy model that catches almost no one.
    best = max(results, key=lambda r: r["roc_auc"])
    screening = max(results, key=lambda r: (r["recall"], r["roc_auc"]))
    chart_confusion_matrix(np.array(screening["confusion_matrix"]), screening["model"])

    imp_pipe = fitted[screening["model"]]
    feature_names = imp_pipe.named_steps["pre"].get_feature_names_out()
    chart_feature_importance(imp_pipe, feature_names, screening["model"])

    summary = {
        "dataset_rows_full": full_rows,
        "sample_used_for_modeling": int(len(df_model)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_full_dataset": full_pos_rate,
        "results": results,
        "best_model_by_roc_auc": best["model"],
        "screening_model_highest_recall": screening["model"],
    }

    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "metrics.txt", "w") as f:
        f.write("Diabetes 30-day readmission prediction\n")
        f.write(f"Full dataset: {summary['dataset_rows_full']:,} encounters\n")
        f.write(f"Positive rate (<30d readmission): "
                f"{summary['positive_rate_full_dataset']:.1%}\n")
        f.write(f"Modeling sample: {summary['sample_used_for_modeling']:,} "
                f"(train {summary['train_rows']:,}, test {summary['test_rows']:,})\n\n")
        for r in results:
            f.write(f"{r['model']}: accuracy={r['accuracy']}, "
                    f"precision={r['precision']}, recall={r['recall']}, "
                    f"F1={r['f1']}, ROC-AUC={r['roc_auc']}\n")
            f.write(f"  confusion matrix (rows=actual No/Yes, cols=pred No/Yes): "
                    f"{r['confusion_matrix']}\n")
        f.write(f"\nBest by ROC-AUC: {summary['best_model_by_roc_auc']}\n")
        f.write(f"Screening model (highest recall): "
                f"{summary['screening_model_highest_recall']}\n")
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
