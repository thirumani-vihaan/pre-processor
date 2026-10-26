from dataclasses import dataclass
import csv
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification


class ValidationError(ValueError):
    """An actionable dataset or experiment configuration error."""


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    name: str
    notes: str
    source_rows: pd.Series | None = None


CATALOG = {
    "demo": {
        "name": "Messy customer signals",
        "description": "Synthetic binary classification with mixed features, missing values, duplicates, and injected feature anomalies. No personal data.",
    },
    "breast_cancer": {
        "name": "Wisconsin breast cancer",
        "description": "569 records, 30 numeric features. Educational benchmark only; not a clinical tool.",
    },
    "wine": {
        "name": "Wine recognition",
        "description": "178 records, 13 numeric features, three cultivar classes.",
    },
    "iris": {
        "name": "Iris flowers",
        "description": "150 records, four numeric features, three species.",
    },
}


def builtin_dataset(key: str, seed: int = 42, missing: float = 0, anomalies: float = 0) -> Dataset:
    if not isinstance(key, str) or key not in CATALOG:
        raise ValidationError("Choose a listed built-in dataset or upload a CSV.")
    rng = np.random.default_rng(seed)
    if key == "demo":
        values, target = make_classification(
            n_samples=720, n_features=8, n_informative=5, n_redundant=1,
            class_sep=1.15, flip_y=0.035, random_state=seed,
        )
        X = pd.DataFrame(values, columns=[
            "engagement", "tenure", "activity", "spend", "visits", "support", "loyalty", "recency",
        ])
        X["channel"] = rng.choice(["Web", "Store", "App"], len(X))
        X["region"] = rng.choice(["North", "South", "East", "West"], len(X))
        y = pd.Series(np.where(target == 1, "Retained", "Churned"), name="class")
        missing = max(missing, 0.035)
        anomalies = max(anomalies, 0.06)
    else:
        source = {
            "breast_cancer": load_breast_cancer,
            "wine": load_wine,
            "iris": load_iris,
        }[key]()
        X = pd.DataFrame(source.data, columns=source.feature_names)
        y = pd.Series(source.target_names[source.target], name="class")
    numeric = X.select_dtypes(include="number").columns
    anomaly_rows = rng.choice(len(X), int(len(X) * anomalies), replace=False)
    if len(anomaly_rows):
        col_positions = [X.columns.get_loc(c) for c in numeric]
        for row in anomaly_rows:
            selected = rng.choice(col_positions, min(3, len(col_positions)), replace=False)
            X.iloc[row, selected] += rng.choice([-1, 1], len(selected)) * (
                X.iloc[:, selected].std().to_numpy() * rng.uniform(6, 12)
            )
    if missing:
        for col in X.columns:
            X.loc[rng.random(len(X)) < missing, col] = np.nan
    if key == "demo":
        X = pd.concat([X, X.iloc[:12]], ignore_index=True)
        y = pd.concat([y, y.iloc[:12]], ignore_index=True)
    notes = (
        f"Seed {seed}. Injected feature anomalies in {len(anomaly_rows)} source rows; "
        f"feature-wise missingness probability {missing:.1%}. Labels are never corrupted. "
        "Injected anomalies are stress-test perturbations, not verified real-world outliers."
    )
    return Dataset(X, y, CATALOG[key]["name"], notes)


def read_csv(content: bytes) -> pd.DataFrame:
    if len(content) > 8 * 1024 * 1024:
        raise ValidationError("CSV exceeds the 8 MB limit.")
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text)), [])
        if not header or any(not name.strip() for name in header):
            raise ValidationError("Give every CSV column a nonblank name.")
        if len(set(header)) != len(header):
            raise ValidationError("CSV column names must be unique; rename duplicate headers.")
        frame = pd.read_csv(BytesIO(content), nrows=5002, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError("Use a comma-separated UTF-8 CSV with a header row.") from exc
    if len(frame) > 5000 or len(frame.columns) > 61:
        raise ValidationError("Use at most 5,000 rows and 60 input features plus one target.")
    if frame.empty or len(frame.columns) < 2:
        raise ValidationError("CSV needs records, at least one feature, and a target column.")
    if any(str(c).startswith("Unnamed:") for c in frame.columns):
        raise ValidationError("Give every column a name; remove unnamed exported index columns.")
    return frame


def csv_dataset(frame: pd.DataFrame, target: str, exclude: list[str]) -> Dataset:
    if not isinstance(target, str) or target not in frame.columns:
        raise ValidationError("Select a target column present in the CSV.")
    if target in exclude or any(col not in frame.columns for col in exclude):
        raise ValidationError("Excluded columns must exist and cannot include the target.")
    X = frame.drop(columns=[target, *exclude])
    if X.shape[1] == 0:
        raise ValidationError("Keep at least one input feature.")
    return Dataset(X.copy(), frame[target].copy(), "Uploaded CSV", "User-supplied data; no synthetic changes applied.")


def prepare(dataset: Dataset) -> tuple[Dataset, dict]:
    X, y = dataset.X.copy(), dataset.y.copy()
    source_rows = pd.Series(
        np.arange(len(X)) if dataset.source_rows is None else dataset.source_rows.to_numpy(),
        index=X.index,
    )
    original = len(X)
    y = y.map(lambda value: value.strip() if isinstance(value, str) else value).replace("", np.nan)
    has_target = y.notna()
    dropped_targets = int((~has_target).sum())
    X, y = X.loc[has_target], y.loc[has_target].astype(str)
    source_rows = source_rows.loc[has_target]
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = X[col].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        X[col] = X[col].replace("", np.nan)
    if X.columns.duplicated().any():
        raise ValidationError("Feature column names must be unique.")
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValidationError(f"Remove columns containing no usable values: {', '.join(all_missing)}.")
    combined = X.copy()
    target_key = "__lab_target__"
    while target_key in combined.columns:
        target_key += "_"
    combined[target_key] = y
    same_features = X.duplicated(keep=False)
    if same_features.any():
        groups = combined.loc[same_features].groupby(list(X.columns), dropna=False, observed=True)[target_key]
        if (groups.nunique() > 1).any():
            raise ValidationError("Identical feature rows have conflicting target labels. Resolve these labels first.")
    duplicate_mask = X.duplicated(keep="first")
    duplicates = int(duplicate_mask.sum())
    X, y = X.loc[~duplicate_mask].reset_index(drop=True), y.loc[~duplicate_mask].reset_index(drop=True)
    source_rows = source_rows.loc[~duplicate_mask].reset_index(drop=True)
    if len(X) < 50:
        raise ValidationError("Provide at least 50 distinct labeled records.")
    counts = y.value_counts()
    if not 2 <= len(counts) <= 15:
        raise ValidationError("Target must contain 2–15 classes; continuous regression targets are not supported.")
    if counts.min() < 8:
        raise ValidationError("Each class needs at least eight distinct records for stratified evaluation.")
    high_cardinality = [
        str(col) for col in X.select_dtypes(exclude="number").columns
        if X[col].nunique() > 30
    ]
    if high_cardinality:
        raise ValidationError(
            f"Exclude identifier/free-text columns with over 30 categories: {', '.join(high_cardinality)}."
        )
    numeric = X.select_dtypes(include="number").columns.tolist()
    profile = {
        "original_rows": original, "rows": len(X), "features": X.shape[1],
        "numeric_features": len(numeric), "categorical_features": X.shape[1] - len(numeric),
        "missing_cells": int(X.isna().sum().sum()), "duplicates_removed": duplicates,
        "missing_targets_removed": dropped_targets,
        "classes": {str(key): int(value) for key, value in counts.items()},
        "columns": [
            {"name": str(col), "type": "numeric" if col in numeric else "categorical",
             "missing": int(X[col].isna().sum()), "unique": int(X[col].nunique())}
            for col in X.columns
        ],
    }
    return Dataset(X, y, dataset.name, dataset.notes, source_rows), profile
