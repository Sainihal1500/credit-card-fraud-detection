from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import importlib.util

import numpy as np
import pandas as pd
from joblib import dump
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

if importlib.util.find_spec("xgboost") is not None:
    XGBClassifier = importlib.import_module("xgboost").XGBClassifier
    XGBOOST_AVAILABLE = True
else:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42
DEFAULT_MAX_ROWS = 50_000
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_SMOTE_RATIO = 0.10
DEFAULT_PARALLEL_JOBS = 1

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("fraud_ml")


IRRELEVANT_COLUMN_PATTERNS = (
    r"^unnamed",
    r".*_id$",
    r"^id$",
    r"cc_num",
    r"trans_num",
    r"first",
    r"last",
    r"street",
    r"address",
    r"full_name",
)

DATE_COLUMN_CANDIDATES = (
    "trans_date_trans_time",
    "transaction_date",
    "transaction_time",
    "timestamp",
    "date",
)

DOB_COLUMN_CANDIDATES = ("dob", "date_of_birth", "birth_date")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create reproducible, reusable fraud-oriented features from a DataFrame."""

    def __init__(self, target_col: str | None = None):
        self.target_col = target_col
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()

        for column in list(frame.columns):
            if _matches_any_pattern(column, IRRELEVANT_COLUMN_PATTERNS):
                frame = frame.drop(columns=[column])

        transaction_dt = _extract_first_datetime_column(frame, DATE_COLUMN_CANDIDATES)

        if transaction_dt is not None:
            frame["trans_hour"] = transaction_dt.dt.hour.astype("float32")
            frame["trans_dayofweek"] = transaction_dt.dt.dayofweek.astype("float32")
            frame["trans_month"] = transaction_dt.dt.month.astype("float32")
            frame["is_night"] = ((transaction_dt.dt.hour < 6) | (transaction_dt.dt.hour >= 20)).astype("float32")
            frame["is_weekend"] = (transaction_dt.dt.dayofweek >= 5).astype("float32")

        dob_dt = _extract_first_datetime_column(frame, DOB_COLUMN_CANDIDATES)
        if dob_dt is not None:
            reference_dt = transaction_dt if transaction_dt is not None else pd.Series(
                pd.Timestamp("2025-01-01"), index=frame.index
            )
            age_years = (reference_dt - dob_dt).dt.days / 365.25
            frame["customer_age"] = age_years.clip(lower=0, upper=120).astype("float32")

        if "amt" in frame.columns:
            amt = pd.to_numeric(frame["amt"], errors="coerce").clip(lower=0)
            frame["log_amt"] = np.log1p(amt)
            frame["amt_squared"] = np.square(amt)

        if "city_pop" in frame.columns and "amt" in frame.columns:
            city_pop = pd.to_numeric(frame["city_pop"], errors="coerce")
            frame["amt_to_city_pop"] = pd.to_numeric(frame["amt"], errors="coerce") / (city_pop + 1.0)

        if {"lat", "long", "merch_lat", "merch_long"}.issubset(frame.columns):
            frame["merchant_distance_km"] = _haversine_km(
                pd.to_numeric(frame["lat"], errors="coerce"),
                pd.to_numeric(frame["long"], errors="coerce"),
                pd.to_numeric(frame["merch_lat"], errors="coerce"),
                pd.to_numeric(frame["merch_long"], errors="coerce"),
            )

        for date_column in list(DATE_COLUMN_CANDIDATES) + list(DOB_COLUMN_CANDIDATES):
            if date_column in frame.columns:
                frame = frame.drop(columns=[date_column])

        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame


@dataclass
class EvalResult:
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    confusion_matrix: list[list[int]]


@dataclass
class RunSummary:
    dataset_shape: tuple[int, int]
    fraud_rate: float
    missing_values: int
    class_distribution: dict[str, int]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    random_forest_results: list[EvalResult]
    hist_gradient_boosting_results: list[EvalResult]
    advanced_model_results: list[EvalResult]
    isolation_forest_results: list[EvalResult]
    all_results_table: list[dict[str, Any]]
    top_feature_importance: list[dict[str, float]]
    eda_summary: dict[str, Any]
    approach_justification: str
    performance_analysis: str
    best_model_by_f1: str
    best_threshold: float
    best_model_path: str
    imbalance_strategy: str
    threshold_metric: str


def _matches_any_pattern(column_name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, str(column_name), flags=re.IGNORECASE) for pattern in patterns)


def _extract_first_datetime_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for candidate in candidates:
        if candidate in frame.columns:
            parsed = pd.to_datetime(frame[candidate], errors="coerce")
            if parsed.notna().any():
                return parsed
    return None


def _haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius_km = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    return pd.Series(radius_km * c, index=lat1.index)


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def stratified_limit_rows(data: pd.DataFrame, target_col: str, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(data) <= max_rows:
        return data.reset_index(drop=True)

    if target_col not in data.columns:
        return data.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    value_counts = data[target_col].value_counts(normalize=True, dropna=False)
    desired = (value_counts * max_rows).round().astype(int)
    desired = desired.clip(lower=1)

    difference = max_rows - int(desired.sum())
    if difference != 0:
        ordered_labels = list(desired.sort_values(ascending=difference > 0).index)
        idx = 0
        while difference != 0 and ordered_labels:
            label = ordered_labels[idx % len(ordered_labels)]
            if difference > 0:
                desired.loc[label] += 1
                difference -= 1
            else:
                if desired.loc[label] > 1:
                    desired.loc[label] -= 1
                    difference += 1
            idx += 1

    sampled_parts: list[pd.DataFrame] = []
    for label, n_samples in desired.items():
        subset = data[data[target_col] == label]
        sampled_parts.append(subset.sample(n=min(len(subset), int(n_samples)), random_state=RANDOM_STATE))

    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def load_dataset(
    data_path: str | None,
    target_col: str,
    demo: bool = False,
    max_rows: int | None = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    if demo:
        return _build_demo_dataset(target_col=target_col, n_rows=8000)

    if data_path is None:
        raise ValueError("Pass --data-path for real dataset or use --demo")

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    else:
        raise ValueError("Supported formats: .csv, .parquet")

    return stratified_limit_rows(data, target_col=target_col, max_rows=max_rows)


def _build_demo_dataset(target_col: str, n_rows: int = 8000) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=n_rows,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.992, 0.008],
        class_sep=1.3,
        random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    base = pd.DataFrame(features, columns=[f"feature_{idx}" for idx in range(features.shape[1])])
    base["trans_date_trans_time"] = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    base["dob"] = pd.to_datetime("1985-01-01") + pd.to_timedelta(rng.integers(0, 12000, size=n_rows), unit="D")
    base["amt"] = np.exp(rng.normal(3.0, 0.8, size=n_rows))
    base["city_pop"] = rng.integers(250, 500000, size=n_rows)
    base["lat"] = rng.uniform(25, 49, size=n_rows)
    base["long"] = rng.uniform(-124, -67, size=n_rows)
    base["merch_lat"] = base["lat"] + rng.normal(0, 0.4, size=n_rows)
    base["merch_long"] = base["long"] + rng.normal(0, 0.4, size=n_rows)
    base["merchant"] = rng.choice(["A", "B", "C", "D", "E"], size=n_rows)
    base["category"] = rng.choice(["food", "travel", "shopping", "gas", "entertainment"], size=n_rows)
    base["gender"] = rng.choice(["M", "F"], size=n_rows)
    base["first"] = rng.choice(["Alex", "Sam", "Jordan", "Taylor"], size=n_rows)
    base["last"] = rng.choice(["Smith", "Lee", "Brown", "Patel"], size=n_rows)
    base["street"] = rng.choice(["Main St", "Oak Ave", "Pine Rd"], size=n_rows)
    base["cc_num"] = rng.integers(10**15, 10**16, size=n_rows)
    base["trans_num"] = [f"T{idx:08d}" for idx in range(n_rows)]
    base[target_col] = target
    return base


def preprocess_target(data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in data.columns:
        available = ", ".join(map(str, data.columns[:20]))
        raise ValueError(f"Target column '{target_col}' not found. Available columns include: {available}")

    work = data.copy()
    unnamed_cols = [col for col in work.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        work = work.drop(columns=unnamed_cols)

    y = work[target_col].astype(int)
    X = work.drop(columns=[target_col])
    return X, y


def explore_dataset(data: pd.DataFrame, target_col: str) -> dict[str, Any]:
    class_counts = data[target_col].value_counts(dropna=False).sort_index().to_dict() if target_col in data.columns else {}
    fraud_rate = float(data[target_col].mean()) if target_col in data.columns else float("nan")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

    summary: dict[str, Any] = {
        "shape": list(data.shape),
        "fraud_rate": fraud_rate,
        "class_distribution": {str(key): int(value) for key, value in class_counts.items()},
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "missing_by_column_top10": data.isna().sum().sort_values(ascending=False).head(10).to_dict(),
    }

    if "amt" in data.columns:
        amt = pd.to_numeric(data["amt"], errors="coerce")
        if amt.notna().any():
            summary["amt_summary"] = {
                "min": float(np.nanmin(amt)),
                "p50": float(np.nanmedian(amt)),
                "mean": float(np.nanmean(amt)),
                "p95": float(np.nanpercentile(amt.dropna(), 95)),
                "max": float(np.nanmax(amt)),
            }

    if categorical_columns:
        sample_categorical = categorical_columns[:3]
        summary["categorical_value_counts"] = {
            column: data[column].astype(str).value_counts(dropna=False).head(5).to_dict() for column in sample_categorical
        }

    return summary


def build_approach_justification_text() -> str:
    return (
        "Supervised learning is chosen because fraud detection is usually pattern-based: fraud transactions "
        "follow recurring signals across amount, time, location, merchant type, and customer history. "
        "A labeled dataset lets the model learn those fraud-vs-legit patterns directly, which is why "
        "supervised models such as Random Forest and Gradient Boosting generally outperform pure anomaly detection. "
        "Isolation Forest is retained only as a baseline comparison because it assumes fraudulent transactions are "
        "rare anomalies rather than structured, repeatable patterns."
    )


def build_performance_analysis(summary: RunSummary) -> str:
    table = pd.DataFrame(summary.all_results_table)
    if table.empty:
        return "Performance analysis unavailable because no model results were produced."

    best_row = table.sort_values("f1", ascending=False).iloc[0]
    if best_row["precision"] >= best_row["recall"]:
        balance_note = "The model is more conservative, so it makes fewer false alarms but may miss some fraud cases."
    else:
        balance_note = "The model is more aggressive, so it catches more fraud but produces more false positives."

    return (
        f"Best model by F1 is {best_row['model']} with precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, and F1={best_row['f1']:.4f}. {balance_note} "
        "HistGradientBoosting usually performs best because it captures non-linear interactions between "
        "amount, time, geo-distance, and derived transaction features while remaining efficient on CPU. "
        "Random Forest is a strong baseline but can be less calibrated and may not capture smooth feature interactions as well. "
        "IsolationForest is weak for this assignment because fraud is not purely random noise; it is a class-specific pattern, "
        "so anomaly detection tends to over-flag legitimate transactions and misses structured fraud cases."
    )


def build_feature_insight_text(top_features: list[dict[str, float]]) -> str:
    if not top_features:
        return "No feature-importance output was available for the selected model."
    top_names = [entry.get("feature") for entry in top_features[:5] if entry.get("feature")]
    joined = ", ".join(map(str, top_names))
    return (
        f"The most influential features were {joined}. These usually matter because fraud often concentrates "
        "in unusual amounts, suspicious transaction timing, merchant-location distance, and other engineered context features."
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    max_categories=25,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_scaled_numeric_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    max_categories=25,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    model: Any,
    resampling_strategy: Literal["none", "smote"] = "none",
    scale_numeric: bool = False,
    smote_ratio: float = DEFAULT_SMOTE_RATIO,
    parallel_jobs: int = DEFAULT_PARALLEL_JOBS,
) -> ImbPipeline:
    steps: list[tuple[str, Any]] = [
        ("features", FeatureEngineer()),
        ("preprocess", build_scaled_numeric_preprocessor() if scale_numeric else build_preprocessor()),
    ]
    if resampling_strategy == "smote":
        steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE, k_neighbors=3)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def optimize_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    metric: Literal["f1", "recall"] = "f1",
    min_accuracy: float = 0.90,
) -> tuple[float, float]:
    thresholds = np.unique(np.clip(np.round(probabilities, 6), 0.0, 1.0))
    if len(thresholds) < 25:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        if accuracy < min_accuracy:
            continue
        if metric == "recall":
            score = recall_score(y_true, predictions, zero_division=0)
            tie_breaker = f1_score(y_true, predictions, zero_division=0)
        else:
            score = f1_score(y_true, predictions, zero_division=0)
            tie_breaker = recall_score(y_true, predictions, zero_division=0)

        if score > best_score or (math.isclose(score, best_score) and tie_breaker > 0):
            best_score = score
            best_threshold = float(threshold)

    if best_score < 0:
        fallback_threshold = 0.5
        return fallback_threshold, float((probabilities >= fallback_threshold).mean())

    return best_threshold, best_score


def evaluate_predictions(
    model_name: str,
    threshold: float,
    y_true: pd.Series,
    scores: np.ndarray,
    positive_label: int = 1,
) -> EvalResult:
    predictions = (scores >= threshold).astype(int)
    if positive_label != 1:
        predictions = np.where(predictions == 1, positive_label, 1 - positive_label)

    return EvalResult(
        model_name=model_name,
        threshold=float(threshold),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        accuracy=float(accuracy_score(y_true, predictions)),
        roc_auc=_safe_roc_auc(y_true, scores),
        confusion_matrix=confusion_matrix(y_true, predictions).tolist(),
    )


def _safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _train_test_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_valid, y_train, y_valid


def _fit_pipeline_with_threshold(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    threshold_metric: Literal["f1", "recall"],
    min_accuracy: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[ImbPipeline, float, float]:
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight

    pipeline.fit(X_train, y_train, **fit_kwargs)
    valid_scores = _predict_scores(pipeline, X_valid)
    threshold, validation_score = optimize_threshold(y_valid, valid_scores, metric=threshold_metric, min_accuracy=min_accuracy)
    return pipeline, threshold, validation_score


def _predict_scores(pipeline: ImbPipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    raise ValueError("Model does not support probability or decision scores")


def train_random_forest_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_metric: Literal["f1", "recall"],
    min_accuracy: float,
    smote_ratio: float,
    parallel_jobs: int,
) -> tuple[list[EvalResult], dict[str, Any]]:
    X_fit, X_valid, y_fit, y_valid = _train_test_validation_split(X_train, y_train, DEFAULT_VALIDATION_SIZE)
    results: list[EvalResult] = []
    artifacts: dict[str, Any] = {}

    baseline = build_model_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, n_jobs=parallel_jobs),
        resampling_strategy="none",
        scale_numeric=False,
    )
    baseline.fit(X_fit, y_fit)
    baseline_threshold, _ = optimize_threshold(y_valid, _predict_scores(baseline, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    baseline_scores = _predict_scores(baseline, X_test)
    results.append(evaluate_predictions("RandomForest baseline", baseline_threshold, y_test, baseline_scores))
    artifacts["RandomForest baseline"] = {"pipeline": baseline, "threshold": baseline_threshold}

    tuned_rf = build_model_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=parallel_jobs),
        resampling_strategy="none",
        scale_numeric=False,
    )
    rf_search = RandomizedSearchCV(
        tuned_rf,
        param_distributions={
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7],
        },
        n_iter=6,
        scoring="f1",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=parallel_jobs,
        verbose=0,
    )
    rf_search.fit(X_fit, y_fit)
    rf_threshold, _ = optimize_threshold(y_valid, _predict_scores(rf_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    rf_scores = _predict_scores(rf_search.best_estimator_, X_test)
    results.append(evaluate_predictions("RandomForest tuned + class_weight", rf_threshold, y_test, rf_scores))
    artifacts["RandomForest tuned + class_weight"] = {"pipeline": rf_search.best_estimator_, "threshold": rf_threshold}

    smote_rf = build_model_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=parallel_jobs),
        resampling_strategy="smote",
        scale_numeric=False,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )
    smote_rf_search = RandomizedSearchCV(
        smote_rf,
        param_distributions={
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7],
        },
        n_iter=5,
        scoring="f1",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=parallel_jobs,
        verbose=0,
    )
    smote_rf_search.fit(X_fit, y_fit)
    smote_rf_threshold, _ = optimize_threshold(y_valid, _predict_scores(smote_rf_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    smote_rf_scores = _predict_scores(smote_rf_search.best_estimator_, X_test)
    results.append(evaluate_predictions("RandomForest tuned + SMOTE", smote_rf_threshold, y_test, smote_rf_scores))
    artifacts["RandomForest tuned + SMOTE"] = {"pipeline": smote_rf_search.best_estimator_, "threshold": smote_rf_threshold}

    return results, artifacts


def train_hist_gradient_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_metric: Literal["f1", "recall"],
    min_accuracy: float,
    smote_ratio: float,
    parallel_jobs: int,
) -> tuple[list[EvalResult], dict[str, Any]]:
    X_fit, X_valid, y_fit, y_valid = _train_test_validation_split(X_train, y_train, DEFAULT_VALIDATION_SIZE)
    results: list[EvalResult] = []
    artifacts: dict[str, Any] = {}

    baseline = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True),
        resampling_strategy="none",
        scale_numeric=False,
    )
    baseline.fit(X_fit, y_fit)
    baseline_threshold, _ = optimize_threshold(y_valid, _predict_scores(baseline, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    baseline_scores = _predict_scores(baseline, X_test)
    results.append(evaluate_predictions("HistGradientBoosting baseline", baseline_threshold, y_test, baseline_scores))
    artifacts["HistGradientBoosting baseline"] = {"pipeline": baseline, "threshold": baseline_threshold}

    class_weight_model = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True),
        resampling_strategy="none",
        scale_numeric=False,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    class_weight_search = RandomizedSearchCV(
        class_weight_model,
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [None, 6, 8, 10],
            "model__max_iter": [150, 200, 250],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 0.3],
            "model__max_leaf_nodes": [31, 63, 127],
        },
        n_iter=6,
        scoring="f1",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=parallel_jobs,
        verbose=0,
    )
    class_weight_search.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    cw_threshold, _ = optimize_threshold(y_valid, _predict_scores(class_weight_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    cw_scores = _predict_scores(class_weight_search.best_estimator_, X_test)
    results.append(evaluate_predictions("HistGradientBoosting + class_weight", cw_threshold, y_test, cw_scores))
    artifacts["HistGradientBoosting + class_weight"] = {"pipeline": class_weight_search.best_estimator_, "threshold": cw_threshold}

    smote_model = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True),
        resampling_strategy="smote",
        scale_numeric=False,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )
    smote_search = RandomizedSearchCV(
        smote_model,
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [None, 6, 8, 10],
            "model__max_iter": [150, 200, 250],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 0.3],
            "model__max_leaf_nodes": [31, 63, 127],
        },
        n_iter=6,
        scoring="f1",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=parallel_jobs,
        verbose=0,
    )
    smote_search.fit(X_fit, y_fit)
    smote_threshold, _ = optimize_threshold(y_valid, _predict_scores(smote_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    smote_scores = _predict_scores(smote_search.best_estimator_, X_test)
    results.append(evaluate_predictions("HistGradientBoosting + SMOTE", smote_threshold, y_test, smote_scores))
    artifacts["HistGradientBoosting + SMOTE"] = {"pipeline": smote_search.best_estimator_, "threshold": smote_threshold}

    tuned_model = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True),
        resampling_strategy="smote",
        scale_numeric=False,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )
    tuned_search = RandomizedSearchCV(
        tuned_model,
        param_distributions={
            "model__learning_rate": [0.02, 0.03, 0.05],
            "model__max_depth": [4, 6, 8],
            "model__max_iter": [180, 220, 280],
            "model__min_samples_leaf": [15, 20, 30],
            "model__l2_regularization": [0.0, 0.05, 0.1],
            "model__max_leaf_nodes": [31, 63, 127],
        },
        n_iter=8,
        scoring="f1",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=parallel_jobs,
        verbose=0,
    )
    tuned_search.fit(X_fit, y_fit)
    tuned_threshold, _ = optimize_threshold(y_valid, _predict_scores(tuned_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    tuned_scores = _predict_scores(tuned_search.best_estimator_, X_test)
    results.append(evaluate_predictions("HistGradientBoosting tuned", tuned_threshold, y_test, tuned_scores))
    artifacts["HistGradientBoosting tuned"] = {"pipeline": tuned_search.best_estimator_, "threshold": tuned_threshold}

    return results, artifacts


def train_optional_advanced_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_metric: Literal["f1", "recall"],
    min_accuracy: float,
    smote_ratio: float,
    parallel_jobs: int,
) -> tuple[list[EvalResult], dict[str, Any]]:
    X_fit, X_valid, y_fit, y_valid = _train_test_validation_split(X_train, y_train, DEFAULT_VALIDATION_SIZE)
    results: list[EvalResult] = []
    artifacts: dict[str, Any] = {}

    if XGBOOST_AVAILABLE:
        xgb = build_model_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_estimators=250,
                n_jobs=parallel_jobs,
            ),
            resampling_strategy="smote",
            scale_numeric=False,
            smote_ratio=smote_ratio,
        )
        xgb_search = RandomizedSearchCV(
            xgb,
            param_distributions={
                "model__n_estimators": [200, 250, 300],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
            n_iter=6,
            scoring="f1",
            cv=2,
            random_state=RANDOM_STATE,
            n_jobs=parallel_jobs,
            verbose=0,
        )
        xgb_search.fit(X_fit, y_fit)
        xgb_threshold, _ = optimize_threshold(y_valid, _predict_scores(xgb_search.best_estimator_, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
        xgb_scores = _predict_scores(xgb_search.best_estimator_, X_test)
        results.append(evaluate_predictions("XGBoost (optional)", xgb_threshold, y_test, xgb_scores))
        artifacts["XGBoost (optional)"] = {"pipeline": xgb_search.best_estimator_, "threshold": xgb_threshold}
        return results, artifacts

    fallback = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True, learning_rate=0.03, max_depth=8, max_iter=300),
        resampling_strategy="smote",
        scale_numeric=False,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )
    fallback.fit(X_fit, y_fit)
    fallback_threshold, _ = optimize_threshold(y_valid, _predict_scores(fallback, X_valid), metric=threshold_metric, min_accuracy=min_accuracy)
    fallback_scores = _predict_scores(fallback, X_test)
    results.append(evaluate_predictions("HistGradientBoosting advanced fallback", fallback_threshold, y_test, fallback_scores))
    artifacts["HistGradientBoosting advanced fallback"] = {"pipeline": fallback, "threshold": fallback_threshold}
    return results, artifacts


def train_isolation_forest_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_metric: Literal["f1", "recall"],
    min_accuracy: float,
    parallel_jobs: int,
) -> list[EvalResult]:
    # Kept only as a baseline comparison: fraud is pattern-based and supervised models learn those labels directly.
    # Isolation Forest is useful as a reference, but it usually performs worse because it treats fraud as generic anomaly noise.
    X_fit, X_valid, y_fit, y_valid = _train_test_validation_split(X_train, y_train, DEFAULT_VALIDATION_SIZE)

    pipeline = ImbPipeline(
        steps=[
            ("features", FeatureEngineer()),
            (
                "preprocess",
                build_scaled_numeric_preprocessor(),
            ),
            ("model", IsolationForest(n_estimators=200, contamination=max(float(y_fit.mean()), 0.001), random_state=RANDOM_STATE, n_jobs=parallel_jobs)),
        ]
    )
    pipeline.fit(X_fit, y_fit)

    valid_scores = -pipeline.named_steps["model"].score_samples(pipeline[:-1].transform(X_valid))
    threshold, _ = optimize_threshold(y_valid, valid_scores, metric=threshold_metric, min_accuracy=min_accuracy)
    test_scores = -pipeline.named_steps["model"].score_samples(pipeline[:-1].transform(X_test))
    return [evaluate_predictions("IsolationForest baseline", threshold, y_test, test_scores)]


def extract_feature_importance(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 12) -> list[dict[str, float]]:
    if estimator is None or not hasattr(estimator, "named_steps"):
        return []

    feature_step = estimator.named_steps.get("features")
    preprocess_step = estimator.named_steps.get("preprocess")
    model_step = estimator.named_steps.get("model")
    if feature_step is None or preprocess_step is None or model_step is None:
        return []

    transformed = preprocess_step.transform(feature_step.transform(X_test))
    feature_names = preprocess_step.get_feature_names_out()

    entries: list[dict[str, float]] = []
    if hasattr(model_step, "feature_importances_"):
        importance = model_step.feature_importances_
        ranking = sorted(zip(feature_names, importance), key=lambda item: item[1], reverse=True)
        entries.extend({"feature": name, "importance": float(value)} for name, value in ranking[:top_n])

    sample_size = min(1500, len(X_test))
    if sample_size >= 50:
        sample_x = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
        sample_y = y_test.loc[sample_x.index]
        permutation = permutation_importance(
            estimator,
            sample_x,
            sample_y,
            scoring="f1",
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        ranking = sorted(zip(feature_names, permutation.importances_mean), key=lambda item: item[1], reverse=True)
        entries.extend({"feature": name, "permutation_f1_drop": float(value)} for name, value in ranking[:top_n])

    return entries


def _results_table(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": result.model_name,
                "threshold": round(result.threshold, 4),
                "precision": round(result.precision, 4),
                "recall": round(result.recall, 4),
                "f1": round(result.f1, 4),
                "accuracy": round(result.accuracy, 4),
                "roc_auc": round(result.roc_auc, 4),
                "tn": result.confusion_matrix[0][0],
                "fp": result.confusion_matrix[0][1],
                "fn": result.confusion_matrix[1][0],
                "tp": result.confusion_matrix[1][1],
            }
            for result in results
        ]
    )


def run_pipeline(
    data_path: str | None,
    target_col: str,
    test_size: float,
    demo: bool = False,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    imbalance_strategy: Literal["class_weight", "smote"] = "smote",
    threshold_metric: Literal["f1", "recall"] = "f1",
    min_accuracy: float = 0.90,
    smote_ratio: float = DEFAULT_SMOTE_RATIO,
    model_output_path: str = "fraud_ml/best_model.joblib",
    parallel_jobs: int = DEFAULT_PARALLEL_JOBS,
) -> RunSummary:
    logger.info("[1/6] Loading data")
    data = load_dataset(data_path=data_path, target_col=target_col, demo=demo, max_rows=max_rows)
    eda_summary = explore_dataset(data, target_col=target_col)
    class_distribution = {
        str(key): int(value)
        for key, value in eda_summary.get("class_distribution", {}).items()
    }
    X, y = preprocess_target(data, target_col=target_col)

    logger.info("[2/6] Creating train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logger.info("[3/6] Training Random Forest models")
    rf_results, rf_artifacts = train_random_forest_models(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_metric=threshold_metric,
        min_accuracy=min_accuracy,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )

    logger.info("[4/6] Training HistGradientBoosting models")
    hgb_results, hgb_artifacts = train_hist_gradient_boosting_models(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_metric=threshold_metric,
        min_accuracy=min_accuracy,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )

    logger.info("[5/6] Training optional advanced model and IsolationForest baseline")
    advanced_results, advanced_artifacts = train_optional_advanced_model(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_metric=threshold_metric,
        min_accuracy=min_accuracy,
        smote_ratio=smote_ratio,
        parallel_jobs=parallel_jobs,
    )
    isolation_results = train_isolation_forest_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_metric=threshold_metric,
        min_accuracy=min_accuracy,
        parallel_jobs=parallel_jobs,
    )

    all_results = rf_results + hgb_results + advanced_results + isolation_results
    best_result = max(all_results, key=lambda item: item.f1)

    artifact_lookup = {**rf_artifacts, **hgb_artifacts, **advanced_artifacts}
    best_artifact = artifact_lookup.get(best_result.model_name)
    best_model_path = Path(model_output_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_artifact is not None:
        dump(
            {
                "pipeline": best_artifact["pipeline"],
                "threshold": best_artifact["threshold"],
                "model_name": best_result.model_name,
                "threshold_metric": threshold_metric,
                "imbalance_strategy": imbalance_strategy,
                "features": list(X.columns),
            },
            best_model_path,
        )

    logger.info("[6/6] Extracting feature importance and preparing summary")
    feature_source = artifact_lookup.get(best_result.model_name, {}).get("pipeline")
    top_features = extract_feature_importance(feature_source, X_test, y_test) if feature_source is not None else []

    combined_results = rf_results + hgb_results + advanced_results + isolation_results
    summary = RunSummary(
        dataset_shape=data.shape,
        fraud_rate=float(y.mean()),
        missing_values=int(data.isna().sum().sum()),
        class_distribution=class_distribution,
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        random_forest_results=rf_results,
        hist_gradient_boosting_results=hgb_results,
        advanced_model_results=advanced_results,
        isolation_forest_results=isolation_results,
        all_results_table=_results_table(combined_results).to_dict(orient="records"),
        top_feature_importance=top_features,
        eda_summary=eda_summary,
        approach_justification=build_approach_justification_text(),
        performance_analysis="",
        best_model_by_f1=best_result.model_name,
        best_threshold=best_result.threshold,
        best_model_path=str(best_model_path),
        imbalance_strategy=imbalance_strategy,
        threshold_metric=threshold_metric,
    )

    summary.performance_analysis = build_performance_analysis(summary)

    return summary


def print_summary(summary: RunSummary) -> None:
    print("=" * 90)
    print("Fraud Detection Pipeline Summary")
    print("=" * 90)
    print(f"Dataset shape: {summary.dataset_shape}")
    print(f"Fraud rate: {summary.fraud_rate:.4%}")
    print(f"Missing values: {summary.missing_values}")
    print(f"Class distribution: {summary.class_distribution}")
    print(f"Train shape: {summary.train_shape} | Test shape: {summary.test_shape}")
    print(f"Best model: {summary.best_model_by_f1} | Best threshold: {summary.best_threshold:.4f}")
    print(f"Saved best model: {summary.best_model_path}")
    print("-")

    print("EDA summary:")
    print(f"  Numeric columns: {summary.eda_summary.get('numeric_column_count')}")
    print(f"  Categorical columns: {summary.eda_summary.get('categorical_column_count')}")
    if "amt_summary" in summary.eda_summary:
        print(f"  Amount distribution: {summary.eda_summary['amt_summary']}")
    if "categorical_value_counts" in summary.eda_summary:
        print(f"  Categorical value counts: {summary.eda_summary['categorical_value_counts']}")
    print("-")

    table = pd.DataFrame(summary.all_results_table)
    if not table.empty:
        print(table.to_string(index=False))
    print("-")

    print("Top feature importance / permutation entries:")
    if summary.top_feature_importance:
        for row in summary.top_feature_importance[:12]:
            print(f"  {row}")
    else:
        print("  Feature importance unavailable for the selected model")

    print("-")
    print("Approach justification:")
    print(f"  {summary.approach_justification}")
    print("-")
    print("Performance analysis:")
    print(f"  {summary.performance_analysis}")
    print("-")
    print("Feature insights:")
    print(f"  {build_feature_insight_text(summary.top_feature_importance)}")


def save_summary_json(summary: RunSummary, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(asdict(summary), indent=2, default=str))


def load_best_model(model_path: str) -> dict[str, Any]:
    from joblib import load

    artifact = load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "threshold" not in artifact:
        raise ValueError("Invalid model artifact")
    return artifact


def predict_new_transactions(model_path: str, records: pd.DataFrame) -> pd.DataFrame:
    artifact = load_best_model(model_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    scores = _predict_scores(pipeline, records)
    predictions = (scores >= threshold).astype(int)
    output = records.copy()
    output["fraud_probability"] = scores
    output["fraud_prediction"] = predictions
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-style credit card fraud detection pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV or Parquet dataset")
    parser.add_argument("--target-col", type=str, default="is_fraud", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--demo", action="store_true", help="Use synthetic mixed-type demo dataset")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Cap dataset size for faster Mac runs")
    parser.add_argument("--imbalance-strategy", choices=["class_weight", "smote"], default="smote")
    parser.add_argument("--threshold-metric", choices=["f1", "recall"], default="f1")
    parser.add_argument("--min-accuracy", type=float, default=0.90, help="Minimum accuracy required during threshold search")
    parser.add_argument("--smote-ratio", type=float, default=DEFAULT_SMOTE_RATIO, help="Minority/majority ratio for controlled SMOTE")
    parser.add_argument("--output-json", type=str, default="fraud_ml/results_summary.json")
    parser.add_argument("--model-path", type=str, default="fraud_ml/best_model.joblib")
    parser.add_argument("--parallel-jobs", type=int, default=DEFAULT_PARALLEL_JOBS, help="Parallel jobs used by searches and estimators")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=not args.quiet)
    summary = run_pipeline(
        data_path=args.data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        demo=args.demo,
        max_rows=args.max_rows,
        imbalance_strategy=args.imbalance_strategy,
        threshold_metric=args.threshold_metric,
        min_accuracy=args.min_accuracy,
        smote_ratio=args.smote_ratio,
        model_output_path=args.model_path,
        parallel_jobs=args.parallel_jobs,
    )
    print_summary(summary)
    save_summary_json(summary, args.output_json)
    print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
