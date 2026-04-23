import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.datasets import make_classification


RANDOM_STATE = 42


@dataclass
class EvalResult:
    model_name: str
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
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    random_forest_results: list[EvalResult]
    isolation_forest_results: list[EvalResult]
    advanced_model_results: list[EvalResult]
    top_feature_importance: list[dict[str, float]]
    best_model_by_f1: str


def _safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def load_dataset(
    data_path: str | None,
    target_col: str,
    demo: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if demo:
        x_data, y_data = make_classification(
            n_samples=12000,
            n_features=30,
            n_informative=12,
            n_redundant=10,
            n_classes=2,
            weights=[0.993, 0.007],
            class_sep=1.2,
            random_state=RANDOM_STATE,
        )
        columns = [f"V{i}" for i in range(1, x_data.shape[1] + 1)]
        frame = pd.DataFrame(x_data, columns=columns)
        frame[target_col] = y_data
        return frame

    if data_path is None:
        raise ValueError("Pass --data-path for real dataset or use --demo")

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet"}:
        data = pd.read_parquet(path)
    else:
        raise ValueError("Supported formats: .csv, .parquet")

    if max_rows is not None and len(data) > max_rows:
        if target_col in data.columns:
            data = (
                data.groupby(target_col, group_keys=False)
                .apply(lambda split: split.sample(frac=max_rows / len(data), random_state=RANDOM_STATE))
                .sample(frac=1.0, random_state=RANDOM_STATE)
                .reset_index(drop=True)
            )
        else:
            data = data.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    return data


def preprocess_data(data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")

    work = data.copy()
    unnamed_cols = [column for column in work.columns if str(column).startswith("Unnamed")]
    if unnamed_cols:
        work = work.drop(columns=unnamed_cols)

    y_data = work[target_col].astype(int)
    x_data = work.drop(columns=[target_col])

    if "trans_date_trans_time" in x_data.columns:
        trans_ts = pd.to_datetime(x_data["trans_date_trans_time"], errors="coerce")
        x_data["trans_hour"] = trans_ts.dt.hour
        x_data["trans_dayofweek"] = trans_ts.dt.dayofweek
        x_data["trans_month"] = trans_ts.dt.month

    if "dob" in x_data.columns:
        dob_ts = pd.to_datetime(x_data["dob"], errors="coerce")
        if "trans_date_trans_time" in x_data.columns:
            trans_ts = pd.to_datetime(x_data["trans_date_trans_time"], errors="coerce")
            age_years = (trans_ts - dob_ts).dt.days / 365.25
            x_data["customer_age"] = age_years

    return x_data, y_data


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


def choose_threshold_with_accuracy_constraint(
    y_true: pd.Series,
    scores: np.ndarray,
    min_accuracy: float = 0.90,
    score_type: str = "proba",
) -> float:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 40)
    best_threshold = float(np.median(thresholds))
    best_f1 = -1.0

    for threshold in thresholds:
        if score_type == "proba":
            preds = (scores >= threshold).astype(int)
        else:
            preds = (scores >= threshold).astype(int)

        accuracy = accuracy_score(y_true, preds)
        if accuracy < min_accuracy:
            continue

        f1_val = f1_score(y_true, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_threshold = float(threshold)

    if best_f1 < 0:
        return float(np.median(thresholds))
    return best_threshold


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=_safe_roc_auc(y_true, y_scores),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def train_random_forest_variants(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[list[EvalResult], Any]:
    results: list[EvalResult] = []

    base_pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=250,
                    max_depth=None,
                    class_weight=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    base_pipeline.fit(x_train, y_train)
    base_proba = base_pipeline.predict_proba(x_test)[:, 1]
    base_pred = (base_proba >= 0.5).astype(int)
    results.append(evaluate_predictions("RF Baseline (No balancing)", y_test, base_pred, base_proba))

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    tuned_pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 16, 24],
        "model__min_samples_split": [2, 4, 8],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.7],
    }

    tuned_search = RandomizedSearchCV(
        estimator=tuned_pipeline,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    tuned_search.fit(x_train, y_train)

    tuned_proba_train = tuned_search.best_estimator_.predict_proba(x_train)[:, 1]
    threshold = choose_threshold_with_accuracy_constraint(y_train, tuned_proba_train, min_accuracy=0.90)
    tuned_proba_test = tuned_search.best_estimator_.predict_proba(x_test)[:, 1]
    tuned_pred_test = (tuned_proba_test >= threshold).astype(int)
    results.append(evaluate_predictions("RF Tuned + class_weight", y_test, tuned_pred_test, tuned_proba_test))

    balanced_pipeline = ImbPipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=False)),
            ("balance", SMOTEENN(random_state=RANDOM_STATE)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    balanced_search = RandomizedSearchCV(
        estimator=balanced_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    balanced_search.fit(x_train, y_train)

    bal_proba_train = balanced_search.best_estimator_.predict_proba(x_train)[:, 1]
    bal_threshold = choose_threshold_with_accuracy_constraint(y_train, bal_proba_train, min_accuracy=0.90)
    bal_proba_test = balanced_search.best_estimator_.predict_proba(x_test)[:, 1]
    bal_pred_test = (bal_proba_test >= bal_threshold).astype(int)
    results.append(
        evaluate_predictions("RF Tuned + SMOTEENN + class_weight", y_test, bal_pred_test, bal_proba_test)
    )

    calibrated = CalibratedClassifierCV(
        estimator=balanced_search.best_estimator_,
        method="sigmoid",
        cv=2,
    )
    calibrated.fit(x_train, y_train)
    cal_train_proba = calibrated.predict_proba(x_train)[:, 1]
    cal_threshold = choose_threshold_with_accuracy_constraint(y_train, cal_train_proba, min_accuracy=0.90)
    cal_test_proba = calibrated.predict_proba(x_test)[:, 1]
    cal_test_pred = (cal_test_proba >= cal_threshold).astype(int)
    results.append(
        evaluate_predictions(
            "RF Tuned + SMOTEENN + Calibrated",
            y_test,
            cal_test_pred,
            cal_test_proba,
        )
    )

    return results, balanced_search.best_estimator_


def train_isolation_forest_variants(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    prep = build_preprocessor(scale_numeric=True)
    x_train_scaled = prep.fit_transform(x_train)
    x_test_scaled = prep.transform(x_test)

    fraud_rate = max(float(y_train.mean()), 0.001)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_scaled,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    baseline_if = IsolationForest(
        n_estimators=220,
        contamination=fraud_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_if.fit(x_fit)
    baseline_scores_val = -baseline_if.score_samples(x_val)
    baseline_threshold = choose_threshold_with_accuracy_constraint(
        y_val,
        baseline_scores_val,
        min_accuracy=0.90,
        score_type="anomaly",
    )
    baseline_scores_test = -baseline_if.score_samples(x_test_scaled)
    baseline_preds_test = (baseline_scores_test >= baseline_threshold).astype(int)
    results.append(
        evaluate_predictions("IF Baseline (scaled + threshold tuned)", y_test, baseline_preds_test, baseline_scores_test)
    )

    contamination_grid = sorted(
        {
            round(max(0.001, min(0.45, fraud_rate * 0.5)), 4),
            round(max(0.001, min(0.45, fraud_rate)), 4),
            round(max(0.001, min(0.45, fraud_rate * 1.5)), 4),
            0.01,
            0.02,
            0.03,
        }
    )

    best_contamination = contamination_grid[0]
    best_threshold = baseline_threshold
    best_f1 = -1.0

    for contamination in contamination_grid:
        model = IsolationForest(
            n_estimators=320,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_fit)
        val_scores = -model.score_samples(x_val)
        threshold = choose_threshold_with_accuracy_constraint(
            y_val,
            val_scores,
            min_accuracy=0.90,
            score_type="anomaly",
        )
        val_pred = (val_scores >= threshold).astype(int)
        score = f1_score(y_val, val_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_contamination = contamination
            best_threshold = threshold

    tuned_if = IsolationForest(
        n_estimators=400,
        contamination=best_contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    tuned_if.fit(x_train_scaled)
    score_tuned = -tuned_if.score_samples(x_test_scaled)
    pred_tuned = (score_tuned >= best_threshold).astype(int)
    results.append(
        evaluate_predictions(
            f"IF Tuned (contamination={best_contamination})",
            y_test,
            pred_tuned,
            score_tuned,
        )
    )

    return results


def train_advanced_boosting_variants(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[list[EvalResult], Any]:
    results: list[EvalResult] = []
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    fallback = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_depth=6,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", fallback),
        ]
    )
    pipeline.fit(x_train, y_train)

    train_proba = pipeline.predict_proba(x_train)[:, 1]
    threshold = choose_threshold_with_accuracy_constraint(y_train, train_proba, min_accuracy=0.90)
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    results.append(evaluate_predictions("HistGradientBoosting (Mac-friendly advanced model)", y_test, test_pred, test_proba))
    return results, pipeline


def extract_feature_importance(
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 12,
) -> list[dict[str, float]]:
    if estimator is None:
        return []

    if not hasattr(estimator, "named_steps"):
        return []

    prep = estimator.named_steps.get("prep")
    model = estimator.named_steps.get("model")
    if prep is None or model is None:
        return []

    names = prep.get_feature_names_out().tolist()
    top_imp: list[dict[str, float]] = []
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        ranking = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
        top_imp = [{"feature": name, "importance": float(value)} for name, value in ranking[:top_n]]

    sample_size = min(1500, len(x_test))
    sample_x = x_test.sample(n=sample_size, random_state=RANDOM_STATE)
    sample_y = y_test.loc[sample_x.index]
    perm = permutation_importance(
        estimator,
        sample_x,
        sample_y,
        scoring="f1",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_rank = sorted(
        zip(x_test.columns.tolist(), perm.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    )
    perm_top = [{"feature": name, "permutation_f1_drop": float(value)} for name, value in perm_rank[:top_n]]

    return top_imp + perm_top


def run_pipeline(
    data_path: str | None,
    target_col: str,
    test_size: float,
    demo: bool = False,
    max_rows: int | None = None,
) -> RunSummary:
    data = load_dataset(data_path, target_col=target_col, demo=demo, max_rows=max_rows)
    x_data, y_data = preprocess_data(data, target_col=target_col)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        stratify=y_data,
        random_state=RANDOM_STATE,
    )

    rf_results, best_rf_estimator = train_random_forest_variants(x_train, y_train, x_test, y_test)
    if_results = train_isolation_forest_variants(x_train, y_train, x_test, y_test)
    adv_results, adv_estimator = train_advanced_boosting_variants(x_train, y_train, x_test, y_test)

    feature_importance = extract_feature_importance(adv_estimator or best_rf_estimator, x_test, y_test)

    all_results = rf_results + if_results + adv_results
    best = max(all_results, key=lambda item: item.f1)

    return RunSummary(
        dataset_shape=data.shape,
        fraud_rate=float(y_data.mean()),
        missing_values=int(data.isna().sum().sum()),
        train_shape=x_train.shape,
        test_shape=x_test.shape,
        random_forest_results=rf_results,
        isolation_forest_results=if_results,
        advanced_model_results=adv_results,
        top_feature_importance=feature_importance,
        best_model_by_f1=best.model_name,
    )


def print_summary(summary: RunSummary) -> None:
    print("=" * 80)
    print("Fraud Detection Experiment Summary")
    print("=" * 80)
    print(f"Dataset shape: {summary.dataset_shape}")
    print(f"Fraud rate: {summary.fraud_rate:.4%}")
    print(f"Missing values: {summary.missing_values}")
    print(f"Train shape: {summary.train_shape} | Test shape: {summary.test_shape}")
    print("-" * 80)

    def print_block(title: str, rows: list[EvalResult]) -> None:
        print(title)
        for row in rows:
            print(
                f"  {row.model_name}\n"
                f"    Precision={row.precision:.4f}, Recall={row.recall:.4f}, F1={row.f1:.4f}, "
                f"Accuracy={row.accuracy:.4f}, ROC-AUC={row.roc_auc:.4f}\n"
                f"    Confusion Matrix={row.confusion_matrix}"
            )
        print("-" * 80)

    print_block("Random Forest Variants:", summary.random_forest_results)
    print_block("Isolation Forest Variants:", summary.isolation_forest_results)
    print_block("Advanced Model Variants:", summary.advanced_model_results)

    print("Top feature importance entries:")
    for row in summary.top_feature_importance[:12]:
        print(f"  {row}")
    print("-" * 80)

    print(f"Best model by F1: {summary.best_model_by_f1}")


def save_summary_json(summary: RunSummary, output_path: str) -> None:
    serializable = asdict(summary)
    Path(output_path).write_text(json.dumps(serializable, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit card fraud detection pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV or Parquet dataset")
    parser.add_argument("--target-col", type=str, default="Class", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--demo", action="store_true", help="Use synthetic imbalanced dataset")
    parser.add_argument("--max-rows", type=int, default=50000, help="Optional cap for very large datasets")
    parser.add_argument(
        "--output-json",
        type=str,
        default="fraud_ml/results_summary.json",
        help="Path to write evaluation summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        data_path=args.data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        demo=args.demo,
        max_rows=args.max_rows,
    )
    print_summary(summary)
    save_summary_json(summary, args.output_json)
    print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
