#!/usr/bin/env python3
"""
SparkSync standalone analysis script

Loads the attached SparkSync clinical dataset, performs dataset-specific cleaning,
runs baseline and tuned ML pipelines for:
  1) Multiclass classification: last_elected_program
  2) Regression: coordinator_comfort

It saves:
  - data profile reports
  - model comparison CSV files
  - feature importance CSV + PNG charts
  - missingness and target distribution plots
  - an optional SHAP summary if the shap package is installed

Usage:
    python sparksync_analysis.py --input sparksync_clinical_data_v2_expanded.csv --output sparksync_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    top_k_accuracy_score,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TOOL_LABEL = "SparkSync"
RANDOM_STATE = 42


# -----------------------------
# Utility functions
# -----------------------------

def make_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_feature_names(transformer: ColumnTransformer) -> np.ndarray:
    try:
        return transformer.get_feature_names_out()
    except Exception:
        names: List[str] = []
        for name, trans, cols in transformer.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    trans_names = trans.get_feature_names_out(cols)
                    names.extend(list(trans_names))
                    continue
                except Exception:
                    pass
            names.extend([str(c) for c in cols])
        return np.array(names, dtype=object)


# -----------------------------
# Data loading and cleaning
# -----------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def profile_dataset(df: pd.DataFrame) -> Dict:
    profile = {
        "tool_label": TOOL_LABEL,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_mrn": int(df["mrn"].duplicated().sum()) if "mrn" in df.columns else None,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_counts": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean().mul(100).round(2)).to_dict(),
        "unique_counts": df.nunique(dropna=False).to_dict(),
    }
    return profile


def clean_sparksync(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    cleaned = df.copy()
    cleaning_log = {
        "tool_label": TOOL_LABEL,
        "dropped_identifier_columns": [],
        "dropped_duplicate_columns": [],
        "dropped_empty_columns": [],
        "engineered_columns": [],
    }

    # Drop identifier
    if "mrn" in cleaned.columns:
        cleaned = cleaned.drop(columns=["mrn"])
        cleaning_log["dropped_identifier_columns"].append("mrn")

    # Drop duplicated diagnosis mirrors when present
    duplicate_map = {
        "diagnosis_1": "primary_diagnosis",
        "diagnosis_2": "secondary_diagnosis",
        "diagnosis_3": "tertiary_diagnosis",
    }
    duplicate_drop = []
    for dup_col, original_col in duplicate_map.items():
        if dup_col in cleaned.columns and original_col in cleaned.columns:
            same = cleaned[dup_col].fillna("<NA>").astype(str).equals(cleaned[original_col].fillna("<NA>").astype(str))
            if same:
                duplicate_drop.append(dup_col)
    if duplicate_drop:
        cleaned = cleaned.drop(columns=duplicate_drop)
        cleaning_log["dropped_duplicate_columns"] = duplicate_drop

    # Drop fully empty columns
    empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
    if empty_cols:
        cleaned = cleaned.drop(columns=empty_cols)
        cleaning_log["dropped_empty_columns"] = empty_cols

    # Add missingness indicator for sparse diagnosis_4 and feedback_sentiment
    if "diagnosis_4" in cleaned.columns:
        cleaned["has_diagnosis_4"] = cleaned["diagnosis_4"].notna()
        cleaning_log["engineered_columns"].append("has_diagnosis_4")

    if "feedback_sentiment" in cleaned.columns:
        cleaned["feedback_sentiment_missing"] = cleaned["feedback_sentiment"].isna()
        cleaning_log["engineered_columns"].append("feedback_sentiment_missing")

    # Add dataset-specific interaction features if source columns are present
    if {"level_of_care", "treatment_specialty"}.issubset(cleaned.columns):
        cleaned["care_specialty_combo"] = (
            cleaned["level_of_care"].astype(str) + " | " + cleaned["treatment_specialty"].astype(str)
        )
        cleaning_log["engineered_columns"].append("care_specialty_combo")

    if {"geography", "payer_arrangement"}.issubset(cleaned.columns):
        cleaned["geo_payer_combo"] = (
            cleaned["geography"].astype(str) + " | " + cleaned["payer_arrangement"].astype(str)
        )
        cleaning_log["engineered_columns"].append("geo_payer_combo")

    return cleaned, cleaning_log


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_missingness(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    missing.plot(kind="bar")
    plt.ylabel("Missing Fraction")
    plt.title(f"{TOOL_LABEL} - Top Missing Columns")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_target_distribution(series: pd.Series, output_path: Path, title: str, top_n: int = 20) -> None:
    counts = series.value_counts(dropna=False).head(top_n)
    plt.figure(figsize=(12, 7))
    counts.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_regression_target(series: pd.Series, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(series.dropna(), bins=30)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_model_comparison(df: pd.DataFrame, metric_col: str, output_path: Path, title: str, ascending: bool = False) -> None:
    plot_df = df.sort_values(metric_col, ascending=ascending)
    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"], plot_df[metric_col])
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 20) -> None:
    top = importance_df.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# -----------------------------
# Modeling helpers
# -----------------------------

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def classification_cv_metrics(estimator, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")

    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=1)
    top3 = top_k_accuracy_score(y, proba, k=min(3, proba.shape[1]), labels=np.unique(y))

    return {
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "macro_f1_mean": float(np.mean(cv_results["test_macro_f1"])),
        "macro_f1_std": float(np.std(cv_results["test_macro_f1"])),
        "weighted_f1_mean": float(np.mean(cv_results["test_weighted_f1"])),
        "weighted_f1_std": float(np.std(cv_results["test_weighted_f1"])),
        "top3_accuracy": float(top3),
        "fit_time_mean": float(np.mean(cv_results["fit_time"])),
        "score_time_mean": float(np.mean(cv_results["score_time"])),
    }


def regression_cv_metrics(estimator, X: pd.DataFrame, y: pd.Series, cv: KFold) -> Dict[str, float]:
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
    return {
        "mae_mean": float(-np.mean(cv_results["test_mae"])),
        "mae_std": float(np.std(-cv_results["test_mae"])),
        "rmse_mean": float(-np.mean(cv_results["test_rmse"])),
        "rmse_std": float(np.std(-cv_results["test_rmse"])),
        "r2_mean": float(np.mean(cv_results["test_r2"])),
        "r2_std": float(np.std(cv_results["test_r2"])),
        "fit_time_mean": float(np.mean(cv_results["fit_time"])),
        "score_time_mean": float(np.mean(cv_results["score_time"])),
    }


def get_classification_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "DummyClassifier": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "LogisticRegression": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=2500)),
        ]),
        "RandomForestClassifier": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def get_regression_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "DummyRegressor": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "RandomForestRegressor": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def tune_random_forest_classifier(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1_macro",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


def tune_random_forest_regressor(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt", 0.5],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


def extract_feature_importance(model_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, task: str) -> pd.DataFrame:
    preprocess = model_pipeline.named_steps["preprocess"]
    fitted_model = model_pipeline.named_steps["model"]
    feature_names = safe_feature_names(preprocess)

    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        return fi

    # Fallback to permutation importance on transformed design matrix
    X_transformed = preprocess.transform(X_train)
    scorer = "f1_macro" if task == "classification" else "neg_mean_absolute_error"
    result = permutation_importance(
        fitted_model,
        X_transformed,
        y_train,
        scoring=scorer,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    fi = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def try_shap_summary(model_pipeline: Pipeline, X_train: pd.DataFrame, output_dir: Path, task: str) -> str:
    try:
        import shap  # type: ignore
    except Exception:
        return "SHAP not available. Install shap to generate SHAP plots."

    try:
        preprocess = model_pipeline.named_steps["preprocess"]
        model = model_pipeline.named_steps["model"]
        X_sample = X_train.sample(min(300, len(X_train)), random_state=RANDOM_STATE)
        X_transformed = preprocess.transform(X_sample)
        feature_names = safe_feature_names(preprocess)

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            plt.figure()
            if task == "classification" and isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_transformed, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
            out = output_dir / f"plots/shap_summary_{task}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=180, bbox_inches="tight")
            plt.close()
            return f"Saved SHAP summary to {out}"
        return "SHAP skipped because the chosen model is not tree-based."
    except Exception as e:
        return f"SHAP generation skipped due to error: {e}"


# -----------------------------
# Main workflow
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=f"{TOOL_LABEL} dataset analysis and visualization script")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="sparksync_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    make_output_dir(output_dir)

    # Load + profile + clean
    raw_df = load_dataset(input_path)
    raw_profile = profile_dataset(raw_df)
    cleaned_df, cleaning_log = clean_sparksync(raw_df)
    cleaned_profile = profile_dataset(cleaned_df)

    save_json(raw_profile, output_dir / "tables/raw_profile.json")
    save_json(cleaning_log, output_dir / "tables/cleaning_log.json")
    save_json(cleaned_profile, output_dir / "tables/cleaned_profile.json")
    pd.DataFrame({
        "column": raw_df.columns,
        "missing_pct": raw_df.isna().mean().mul(100).round(2).values,
        "unique_count": raw_df.nunique(dropna=False).values,
        "dtype": [str(t) for t in raw_df.dtypes.values],
    }).sort_values("missing_pct", ascending=False).to_csv(output_dir / "tables/data_profile.csv", index=False)

    # Dataset-level plots
    plot_missingness(raw_df, output_dir / "plots/missingness_raw.png")
    if "last_elected_program" in cleaned_df.columns:
        plot_target_distribution(
            cleaned_df["last_elected_program"],
            output_dir / "plots/last_elected_program_distribution.png",
            f"{TOOL_LABEL} - last_elected_program distribution (top 20)",
        )
    if "coordinator_comfort" in cleaned_df.columns:
        plot_regression_target(
            cleaned_df["coordinator_comfort"],
            output_dir / "plots/coordinator_comfort_distribution.png",
            f"{TOOL_LABEL} - coordinator_comfort distribution",
        )

    summary_lines = []
    summary_lines.append(f"{TOOL_LABEL} analysis summary")
    summary_lines.append(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    summary_lines.append(f"Dropped identifier columns: {cleaning_log['dropped_identifier_columns']}")
    summary_lines.append(f"Dropped duplicate columns: {cleaning_log['dropped_duplicate_columns']}")
    summary_lines.append(f"Dropped empty columns: {cleaning_log['dropped_empty_columns']}")
    summary_lines.append(f"Engineered columns: {cleaning_log['engineered_columns']}")

    # -----------------
    # Classification
    # -----------------
    cls_target = "last_elected_program"
    reg_target = "coordinator_comfort"

    if cls_target not in cleaned_df.columns or reg_target not in cleaned_df.columns:
        raise ValueError("Expected targets last_elected_program and coordinator_comfort were not found.")

    X_cls = cleaned_df.drop(columns=[cls_target, reg_target])
    y_cls = cleaned_df[cls_target]
    pre_cls, _, _ = build_preprocessor(X_cls)
    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cls_models = get_classification_models(pre_cls)
    cls_results = []
    for name, model in cls_models.items():
        metrics = classification_cv_metrics(model, X_cls, y_cls, cv_cls)
        metrics["model"] = name
        cls_results.append(metrics)

    tuned_cls_model = tune_random_forest_classifier(pre_cls, X_cls, y_cls)
    tuned_cls_metrics = classification_cv_metrics(tuned_cls_model, X_cls, y_cls, cv_cls)
    tuned_cls_metrics["model"] = "RandomForestClassifier_Tuned"
    cls_results.append(tuned_cls_metrics)

    cls_results_df = pd.DataFrame(cls_results).sort_values(["macro_f1_mean", "top3_accuracy"], ascending=False)
    cls_results_df.to_csv(output_dir / "tables/classification_results.csv", index=False)
    plot_model_comparison(
        cls_results_df,
        "macro_f1_mean",
        output_dir / "plots/classification_macro_f1_comparison.png",
        f"{TOOL_LABEL} - Classification Macro F1 Comparison",
        ascending=False,
    )
    plot_model_comparison(
        cls_results_df,
        "top3_accuracy",
        output_dir / "plots/classification_top3_comparison.png",
        f"{TOOL_LABEL} - Classification Top-3 Accuracy Comparison",
        ascending=False,
    )

    best_cls_name = cls_results_df.iloc[0]["model"]
    best_cls_model = tuned_cls_model if best_cls_name == "RandomForestClassifier_Tuned" else cls_models[best_cls_name]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=RANDOM_STATE
    )
    best_cls_model.fit(Xc_train, yc_train)
    yc_pred = best_cls_model.predict(Xc_test)
    yc_proba = best_cls_model.predict_proba(Xc_test)
    cls_holdout = {
        "accuracy": float(accuracy_score(yc_test, yc_pred)),
        "macro_f1": float(f1_score(yc_test, yc_pred, average="macro")),
        "weighted_f1": float(f1_score(yc_test, yc_pred, average="weighted")),
        "top3_accuracy": float(top_k_accuracy_score(yc_test, yc_proba, k=min(3, yc_proba.shape[1]), labels=np.unique(y_cls))),
    }
    save_json(cls_holdout, output_dir / "tables/classification_holdout_metrics.json")

    cls_fi = extract_feature_importance(best_cls_model, Xc_train, yc_train, task="classification")
    cls_fi.to_csv(output_dir / "tables/classification_feature_importance.csv", index=False)
    plot_feature_importance(
        cls_fi,
        output_dir / "plots/classification_feature_importance.png",
        f"{TOOL_LABEL} - Classification Feature Importance",
    )
    cls_shap_msg = try_shap_summary(best_cls_model, Xc_train, output_dir, task="classification")

    summary_lines.append("")
    summary_lines.append("Classification best model:")
    summary_lines.append(f"  {best_cls_name}")
    summary_lines.append(f"  CV macro F1: {cls_results_df.iloc[0]['macro_f1_mean']:.4f}")
    summary_lines.append(f"  CV top-3 accuracy: {cls_results_df.iloc[0]['top3_accuracy']:.4f}")
    summary_lines.append(f"  Holdout macro F1: {cls_holdout['macro_f1']:.4f}")

    # -----------------
    # Regression
    # -----------------
    X_reg = cleaned_df.drop(columns=[reg_target, cls_target])
    y_reg = cleaned_df[reg_target]
    pre_reg, _, _ = build_preprocessor(X_reg)
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    reg_models = get_regression_models(pre_reg)
    reg_results = []
    for name, model in reg_models.items():
        metrics = regression_cv_metrics(model, X_reg, y_reg, cv_reg)
        metrics["model"] = name
        reg_results.append(metrics)

    tuned_reg_model = tune_random_forest_regressor(pre_reg, X_reg, y_reg)
    tuned_reg_metrics = regression_cv_metrics(tuned_reg_model, X_reg, y_reg, cv_reg)
    tuned_reg_metrics["model"] = "RandomForestRegressor_Tuned"
    reg_results.append(tuned_reg_metrics)

    reg_results_df = pd.DataFrame(reg_results).sort_values(["mae_mean", "rmse_mean"], ascending=True)
    reg_results_df.to_csv(output_dir / "tables/regression_results.csv", index=False)
    plot_model_comparison(
        reg_results_df,
        "mae_mean",
        output_dir / "plots/regression_mae_comparison.png",
        f"{TOOL_LABEL} - Regression MAE Comparison",
        ascending=True,
    )
    plot_model_comparison(
        reg_results_df,
        "r2_mean",
        output_dir / "plots/regression_r2_comparison.png",
        f"{TOOL_LABEL} - Regression R² Comparison",
        ascending=False,
    )

    best_reg_name = reg_results_df.iloc[0]["model"]
    best_reg_model = tuned_reg_model if best_reg_name == "RandomForestRegressor_Tuned" else reg_models[best_reg_name]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    best_reg_model.fit(Xr_train, yr_train)
    yr_pred = best_reg_model.predict(Xr_test)
    reg_holdout = {
        "mae": float(mean_absolute_error(yr_test, yr_pred)),
        "rmse": float(np.sqrt(mean_squared_error(yr_test, yr_pred))),
        "r2": float(r2_score(yr_test, yr_pred)),
    }
    save_json(reg_holdout, output_dir / "tables/regression_holdout_metrics.json")

    reg_fi = extract_feature_importance(best_reg_model, Xr_train, yr_train, task="regression")
    reg_fi.to_csv(output_dir / "tables/regression_feature_importance.csv", index=False)
    plot_feature_importance(
        reg_fi,
        output_dir / "plots/regression_feature_importance.png",
        f"{TOOL_LABEL} - Regression Feature Importance",
    )
    reg_shap_msg = try_shap_summary(best_reg_model, Xr_train, output_dir, task="regression")

    summary_lines.append("")
    summary_lines.append("Regression best model:")
    summary_lines.append(f"  {best_reg_name}")
    summary_lines.append(f"  CV MAE: {reg_results_df.iloc[0]['mae_mean']:.4f}")
    summary_lines.append(f"  CV R2: {reg_results_df.iloc[0]['r2_mean']:.4f}")
    summary_lines.append(f"  Holdout MAE: {reg_holdout['mae']:.4f}")

    summary_lines.append("")
    summary_lines.append(cls_shap_msg)
    summary_lines.append(reg_shap_msg)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
