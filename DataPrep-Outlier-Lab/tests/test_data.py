import io

import numpy as np
import pandas as pd
import pytest

from lab.data import Dataset, ValidationError, builtin_dataset, csv_dataset, prepare, read_csv
from lab.engine import Config


def test_demo_is_deterministic_and_exercises_cleaning():
    first, second = builtin_dataset("demo"), builtin_dataset("demo")
    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    cleaned, profile = prepare(first)
    assert profile["duplicates_removed"] == 12
    assert profile["missing_cells"] > 0
    assert profile["numeric_features"] == 8
    assert profile["categorical_features"] == 2
    assert len(cleaned.X) == 720


def test_conflicting_duplicate_features_rejected():
    dataset = builtin_dataset("wine")
    dataset.X = pd.concat([dataset.X, dataset.X.iloc[[0]]], ignore_index=True)
    dataset.y = pd.concat([dataset.y, pd.Series(["different"])], ignore_index=True)
    with pytest.raises(ValidationError, match="conflicting"):
        prepare(dataset)


def test_missing_labels_and_nonfinite_values_are_explicit():
    dataset = builtin_dataset("wine")
    dataset.y.iloc[0] = "   "
    dataset.X.iloc[2, 0] = np.inf
    _, profile = prepare(dataset)
    assert profile["missing_targets_removed"] == 1
    assert profile["missing_cells"] == 1


def test_csv_read_and_excluded_identifier():
    source = builtin_dataset("wine")
    frame = source.X.copy()
    frame["record_id"] = [f"record-{i}" for i in range(len(frame))]
    frame["target"] = source.y
    loaded = read_csv(frame.to_csv(index=False).encode())
    with pytest.raises(ValidationError, match="identifier"):
        prepare(csv_dataset(loaded, "target", []))
    _, profile = prepare(csv_dataset(loaded, "target", ["record_id"]))
    assert profile["features"] == 13


@pytest.mark.parametrize("content", [b"", b"just_one_column\nvalue", b"\xff\xfe", b"a,b\n"])
def test_bad_csv_is_actionable(content):
    with pytest.raises(ValidationError):
        read_csv(content)


@pytest.mark.parametrize("settings", [
    {"models": []}, {"models": ["knn", "knn"]}, {"methods": ["unknown"]},
    {"folds": 4}, {"seed": True}, {"seed": -1}, {"test_size": .9},
    {"contamination": float("nan")}, {"pca": "yes"}, {"scaler": "unknown"},
    {"models": [{}]}, {"unknown": 1}, None,
])
def test_bad_configuration_is_actionable(settings):
    with pytest.raises(ValidationError):
        Config.parse(settings)


def test_empty_feature_column_rejected():
    dataset = builtin_dataset("wine")
    dataset.X["empty"] = np.nan
    with pytest.raises(ValidationError, match="no usable"):
        prepare(dataset)


def test_continuous_target_rejected():
    dataset = builtin_dataset("wine")
    dataset.y = pd.Series(np.arange(len(dataset.X)))
    with pytest.raises(ValidationError, match="2–15"):
        prepare(dataset)


def test_too_many_rows_rejected():
    frame = pd.DataFrame({"a": range(5001), "b": range(5001)})
    with pytest.raises(ValidationError, match="5,000"):
        read_csv(frame.to_csv(index=False).encode())


def test_no_features_rejected():
    with pytest.raises(ValidationError, match="at least one"):
        csv_dataset(pd.DataFrame({"target": [0, 1]}), "target", [])


@pytest.mark.parametrize("header", ["feature,feature,label", "  ,feature,label", ",feature,label"])
def test_ambiguous_csv_headers_are_rejected(header):
    with pytest.raises(ValidationError, match="name|header"):
        read_csv(f"{header}\n1,2,a\n3,4,b\n".encode())


def test_cleaning_preserves_original_record_positions():
    dataset = builtin_dataset("wine")
    dataset.y.iloc[0] = np.nan
    dataset.X.iloc[4] = dataset.X.iloc[3]
    dataset.y.iloc[4] = dataset.y.iloc[3]
    cleaned, _ = prepare(dataset)
    assert cleaned.source_rows.iloc[0] == 1
    assert 0 not in cleaned.source_rows.values
    assert 4 not in cleaned.source_rows.values
    again, _ = prepare(cleaned)
    pd.testing.assert_series_equal(cleaned.source_rows, again.source_rows)
