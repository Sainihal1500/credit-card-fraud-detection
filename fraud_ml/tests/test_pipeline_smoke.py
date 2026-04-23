from fraud_ml.fraud_pipeline import run_pipeline


def test_pipeline_demo_runs_and_returns_results():
    summary = run_pipeline(data_path=None, target_col="Class", test_size=0.2, demo=True)

    assert len(summary.random_forest_results) >= 3
    assert len(summary.hist_gradient_boosting_results) >= 3
    assert len(summary.advanced_model_results) >= 1
    assert len(summary.all_results_table) >= 1
    assert summary.best_model_path.endswith(".joblib")
    assert summary.fraud_rate > 0
    assert isinstance(summary.best_model_by_f1, str)
