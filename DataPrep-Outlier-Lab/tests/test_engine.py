import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from lab.data import ValidationError, builtin_dataset, prepare
from lab.engine import Config, detect_outliers, experiment, fit_arm, valid_filter


@pytest.fixture(scope="module")
def result():
    return experiment(builtin_dataset("wine"), Config(models=("nb", "knn"), methods=("iqr", "isolation", "lof", "dbscan")))


def test_all_arms_share_untouched_test_records(result):
    report, _, predictions, audit, processed = result
    split = report["split"]
    assert not set(split["train_ids"]) & set(split["test_ids"])
    assert len(predictions) == split["test"]
    assert len(audit) == split["train"]
    for row in report["results"]:
        assert np.sum(row["confusion"]) == split["test"]
        assert row["removed"] + row["training_rows"] == split["train"]
        assert len(row["fold_scores"]) == report["config"]["folds"]
        assert all(0 <= row[key] <= 1 for key in ("accuracy", "f1", "roc_auc", "cv_f1"))
    selected = next(row for row in report["results"] if row["id"] == report["selected_id"])
    assert len(processed) == selected["training_rows"]
    assert processed.select_dtypes(include="number").notna().all().all()


def test_selection_uses_cv_not_test_accuracy(result):
    report = result[0]
    assert report["selected_id"] == max(report["results"], key=lambda row: row["cv_f1"])["id"]
    selected = next(row for row in report["results"] if row["id"] == report["selected_id"])
    baseline = next(row for row in report["results"] if row["id"] == report["baseline_id"])
    assert selected["model_key"] == baseline["model_key"]
    assert report["accuracy_delta"] == selected["accuracy"] - baseline["accuracy"]


def test_test_feature_changes_do_not_change_training_or_selection():
    dataset, _ = prepare(builtin_dataset("wine"))
    config = Config(models=("nb",), methods=("iqr",), pca=True)
    _, test_ids = train_test_split(np.arange(len(dataset.X)), test_size=config.test_size, stratify=dataset.y, random_state=config.seed)
    first = experiment(dataset, config)
    dataset.X.iloc[test_ids] = dataset.X.iloc[test_ids] + 1000
    second = experiment(dataset, config)
    assert first[0]["selected_id"] == second[0]["selected_id"]
    assert first[0]["outlier_views"] == second[0]["outlier_views"]
    assert [row["cv_f1"] for row in first[0]["results"]] == [row["cv_f1"] for row in second[0]["results"]]
    pd.testing.assert_frame_equal(first[3], second[3])
    pd.testing.assert_frame_equal(first[4], second[4])


def test_imputer_is_fitted_only_on_retained_training():
    dataset, _ = prepare(builtin_dataset("demo"))
    X, y = dataset.X.iloc[:400], dataset.y.iloc[:400]
    pipeline, mask, _, _ = fit_arm(X, y, Config(), "nb", "outlier", "isolation")
    actual = pipeline.named_steps["preprocess"].named_steps["columns"].named_transformers_["numeric"].named_steps["impute"].statistics_
    expected = X.loc[~mask].select_dtypes(include="number").median().to_numpy()
    np.testing.assert_allclose(actual, expected)
    with_unknown = X.iloc[:2].copy()
    with_unknown["channel"] = "Unseen category"
    assert len(pipeline.predict(with_unknown)) == 2


@pytest.mark.parametrize("method", ["iqr", "zscore", "isolation", "lof", "dbscan"])
def test_every_detector_is_repeatable(method):
    dataset, _ = prepare(builtin_dataset("demo"))
    first = detect_outliers(dataset.X, Config(), method)
    second = detect_outliers(dataset.X, Config(), method)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.isfinite(first[1]).all()


def test_excessive_dbscan_filter_is_reported_not_silently_replaced():
    report = experiment(builtin_dataset("wine"), Config(models=("nb",), methods=("dbscan",), dbscan_eps=.1))[0]
    assert report["rejected"]
    assert all(row["arm"] != "outlier" for row in report["results"])
    assert report["best_outlier_id"] is None


def test_minority_class_cannot_be_eliminated():
    y = pd.Series(["a"] * 50 + ["b"] * 10)
    mask = np.array([False] * 50 + [True] * 10)
    assert "class" in valid_filter(mask, y)


def test_categorical_only_outlier_arms_are_explicitly_rejected():
    dataset = builtin_dataset("wine")
    dataset.X = pd.DataFrame({
        "category_a": [str(i // 20) for i in range(len(dataset.X))],
        "category_b": [str(i % 20) for i in range(len(dataset.X))],
    })
    report = experiment(dataset, Config(models=("nb",), methods=("iqr",)))[0]
    assert report["rejected"]
    assert "numeric feature" in report["rejected"][0]["reason"]
    assert len(report["results"]) == 2


def test_per_class_removal_and_distribution_views_agree(result):
    report = result[0]
    for method, view in report["outlier_views"].items():
        assert sum(item["total"] for item in view["classes"]) == report["split"]["train"]
        assert sum(item["removed"] for item in view["classes"]) == view["removed"]
        assert sum(view["mask"]) == view["removed"]
        assert method in report["quantiles_by_method"]


@pytest.mark.parametrize("model", ["tree", "entropy", "nb", "knn", "svm", "rf", "mlp"])
def test_all_classifiers_predict_with_pca(model):
    dataset = builtin_dataset("wine")
    pipeline, _, _, _ = fit_arm(dataset.X, dataset.y, Config(pca=True), model, "preprocessed")
    assert len(pipeline.predict(dataset.X.iloc[:3])) == 3
    assert pipeline.predict_proba(dataset.X.iloc[:3]).shape == (3, 3)


def test_export_metadata_cannot_overwrite_feature_columns():
    dataset = builtin_dataset("iris")
    dataset.X = dataset.X.rename(columns={dataset.X.columns[0]: "source_row", dataset.X.columns[1]: "actual_class"})
    _, _, predictions, audit, _ = experiment(dataset, Config(models=("nb",), methods=("iqr",)))
    assert "feature__source_row" in predictions
    assert "feature__actual_class" in audit


def test_odd_training_size_cannot_lose_more_than_half():
    y = pd.Series(["a", "b"] * 20 + ["a"])
    mask = np.array([False] * 20 + [True] * 21)
    assert valid_filter(mask, y) is not None
    assert valid_filter(np.array([False] * 21 + [True] * 20), y) is None


def test_exports_and_scatter_map_back_to_original_records():
    dataset = builtin_dataset("wine")
    dataset.y.iloc[0] = np.nan
    dataset.X.iloc[4] = dataset.X.iloc[3]
    dataset.y.iloc[4] = dataset.y.iloc[3]
    report, _, predictions, audit, processed = experiment(dataset, Config(models=("nb",), methods=("iqr",)))
    for frame in (predictions, audit):
        ids = frame["source_row"].to_numpy()
        np.testing.assert_allclose(frame["feature__alcohol"], dataset.X.iloc[ids]["alcohol"])
        assert 0 not in ids and 4 not in ids
    assert set(processed["source_row"]) <= set(audit["source_row"])
    assert [point["row"] for point in report["scatter"]] == report["split"]["train_source_rows"]
    assert report["split"]["test_source_rows"] == predictions["source_row"].tolist()


def test_outlier_only_delta_uses_matched_preprocessed_model(result):
    report = result[0]
    outlier = next(row for row in report["results"] if row["id"] == report["best_outlier_id"])
    preprocessed = next(row for row in report["results"] if row["model_key"] == outlier["model_key"] and row["arm"] == "preprocessed")
    assert report["outlier_only_accuracy_delta"] == outlier["accuracy"] - preprocessed["accuracy"]
