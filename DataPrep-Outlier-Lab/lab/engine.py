import warnings
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lab.data import Dataset, ValidationError, prepare


MODEL_NAMES = {
    "tree": "CART decision tree", "entropy": "Entropy decision tree",
    "nb": "Gaussian Naive Bayes", "knn": "K-nearest neighbors",
    "svm": "Support vector machine", "rf": "Random forest", "mlp": "Neural network",
}
METHOD_NAMES = {
    "iqr": "IQR fences", "zscore": "Z-score", "isolation": "Isolation forest",
    "lof": "Local outlier factor", "dbscan": "DBSCAN noise",
}


@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.25
    folds: int = 3
    scaler: str = "robust"
    pca: bool = False
    contamination: float = 0.06
    iqr_factor: float = 2.5
    z_threshold: float = 3.0
    dbscan_eps: float = 2.0
    models: tuple = ("tree", "nb", "knn", "svm", "rf")
    methods: tuple = ("iqr", "zscore", "isolation", "lof", "dbscan")

    @classmethod
    def parse(cls, values: dict):
        allowed = cls.__dataclass_fields__
        if not isinstance(values, dict) or set(values) - set(allowed):
            raise ValidationError("Unknown experiment settings.")
        config = cls(**values)
        for key in ("seed", "folds"):
            if type(getattr(config, key)) is not int:
                raise ValidationError(f"{key} must be an integer.")
        if not 0 <= config.seed <= 2**31 - 1:
            raise ValidationError("Seed must be between 0 and 2147483647.")
        if config.folds not in (3, 5):
            raise ValidationError("Use three or five cross-validation folds.")
        for key, low, high in (
            ("test_size", 0.15, 0.35), ("contamination", 0.01, 0.20),
            ("iqr_factor", 1.5, 5), ("z_threshold", 2, 5), ("dbscan_eps", 0.1, 10),
        ):
            value = getattr(config, key)
            if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
                raise ValidationError(f"{key} must be between {low} and {high}.")
        if config.scaler not in ("standard", "robust", "minmax") or type(config.pca) is not bool:
            raise ValidationError("Select a valid scaler and PCA setting.")
        for key, catalog in (("models", MODEL_NAMES), ("methods", METHOD_NAMES)):
            items = getattr(config, key)
            if not isinstance(items, (list, tuple)) or not items or any(not isinstance(item, str) for item in items):
                raise ValidationError(f"Choose a nonempty list of {key}.")
            if len(set(items)) != len(items):
                raise ValidationError(f"Choose unique {key}; at least one is required.")
            if any(item not in catalog for item in items):
                raise ValidationError(f"Unknown {key} selected.")
        return config


def estimator(key: str, seed: int):
    return {
        "tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=3, random_state=seed),
        "entropy": DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=3, random_state=seed),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=1),
        "svm": SVC(C=1.0, kernel="rbf", probability=True, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "mlp": MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=400, random_state=seed),
    }[key]


def preprocessor(X: pd.DataFrame, config: Config, baseline: bool = False):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if not baseline:
        numeric_steps.append(("scale", {
            "standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(),
        }[config.scaler]))
    transformations = []
    if numeric:
        transformations.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformations.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    steps = [("columns", ColumnTransformer(transformations, sparse_threshold=0))]
    if config.pca and not baseline:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return Pipeline(steps)


def detect_outliers(X: pd.DataFrame, config: Config, method: str) -> tuple[np.ndarray, np.ndarray]:
    numeric = X.select_dtypes(include="number")
    if not len(numeric.columns):
        raise ValidationError("Outlier detection needs at least one numeric feature; categorical columns are still encoded for classification.")
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(numeric)
    scaled = StandardScaler().fit_transform(imputed)
    if method == "iqr":
        q1, q3 = np.quantile(imputed, [0.25, 0.75], axis=0)
        span = q3 - q1
        # A zero-IQR feature can still have rare deviations; retain those signals.
        denominator = np.where(span > 0, span, 1.0)
        scores = np.maximum((q1 - imputed) / denominator, (imputed - q3) / denominator).max(axis=1)
        mask = ((imputed < q1 - config.iqr_factor * span) | (imputed > q3 + config.iqr_factor * span)).any(axis=1)
    elif method == "zscore":
        scores = np.abs(scaled).max(axis=1)
        mask = scores > config.z_threshold
    elif method == "isolation":
        detector = IsolationForest(n_estimators=100, contamination=config.contamination, random_state=config.seed, n_jobs=1)
        mask = detector.fit_predict(scaled) == -1
        scores = -detector.score_samples(scaled)
    elif method == "lof":
        detector = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination=config.contamination, n_jobs=1)
        mask = detector.fit_predict(scaled) == -1
        scores = -detector.negative_outlier_factor_
    elif method == "dbscan":
        labels = DBSCAN(eps=config.dbscan_eps, min_samples=5, n_jobs=1).fit_predict(scaled)
        mask = labels == -1
        scores = mask.astype(float)
    else:
        raise ValidationError(f"Unknown outlier detector: {method}.")
    return np.asarray(mask), np.asarray(scores)


def valid_filter(mask: np.ndarray, y: pd.Series) -> str | None:
    remaining = y.loc[~mask]
    if len(remaining) < 20 or len(remaining) * 2 < len(y):
        return "Detector would remove over half the training rows or leave fewer than 20; adjust its threshold."
    if set(remaining) != set(y) or remaining.value_counts().min() < 3:
        return "Detector would eliminate a class or leave fewer than three class examples."
    return None


def fit_arm(X, y, config, model, arm, method=None):
    mask = np.zeros(len(X), dtype=bool)
    scores = np.zeros(len(X))
    if method:
        mask, scores = detect_outliers(X, config, method)
        problem = valid_filter(mask, y)
        if problem:
            raise ValidationError(problem)
    retained = X.loc[~mask]
    pipeline = Pipeline([
        ("preprocess", preprocessor(retained, config, baseline=arm == "baseline")),
        ("classifier", clone(estimator(model, config.seed))),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(retained, y.loc[~mask])
    messages = sorted({str(w.message) for w in caught})
    return pipeline, mask, scores, messages


def metrics(y, pred, probabilities, classes):
    values = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=classes).tolist(),
    }
    if len(classes) == 2:
        values["roc_auc"] = float(roc_auc_score(np.asarray(y) == classes[1], probabilities[:, 1]))
    else:
        values["roc_auc"] = float(roc_auc_score(y, probabilities, labels=classes, multi_class="ovr", average="macro"))
    return values


def experiment(dataset: Dataset, config: Config):
    started = perf_counter()
    dataset, profile = prepare(dataset)
    X, y = dataset.X, dataset.y
    source_rows = dataset.source_rows.to_numpy()
    train_ids, test_ids = train_test_split(
        np.arange(len(X)), test_size=config.test_size, stratify=y, random_state=config.seed,
    )
    Xtrain, Xtest = X.iloc[train_ids].reset_index(drop=True), X.iloc[test_ids].reset_index(drop=True)
    ytrain, ytest = y.iloc[train_ids].reset_index(drop=True), y.iloc[test_ids].reset_index(drop=True)
    if ytrain.value_counts().min() < config.folds:
        raise ValidationError("Training class counts are too small for the selected fold count.")
    folds = list(StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed).split(Xtrain, ytrain))
    classes = sorted(y.unique())
    results, rejected, warnings_seen = [], [], set()
    fitted = {}
    arms = [("baseline", None), ("preprocessed", None)] + [("outlier", method) for method in config.methods]
    for model in config.models:
        for arm, method in arms:
            key = f"{model}:{method or arm}"
            fold_scores = []
            try:
                for fit_ids, val_ids in folds:
                    fold_X = Xtrain.iloc[fit_ids].reset_index(drop=True)
                    fold_y = ytrain.iloc[fit_ids].reset_index(drop=True)
                    pipeline, _, _, messages = fit_arm(fold_X, fold_y, config, model, arm, method)
                    warnings_seen.update(messages)
                    prediction = pipeline.predict(Xtrain.iloc[val_ids])
                    fold_scores.append(float(f1_score(ytrain.iloc[val_ids], prediction, average="macro", zero_division=0)))
                pipeline, mask, scores, messages = fit_arm(Xtrain, ytrain, config, model, arm, method)
                warnings_seen.update(messages)
            except ValidationError as exc:
                rejected.append({"model": MODEL_NAMES[model], "method": METHOD_NAMES.get(method, arm), "reason": str(exc)})
                continue
            fitted[key] = (pipeline, mask, scores)
            results.append({
                "id": key, "model_key": model, "model": MODEL_NAMES[model], "arm": arm,
                "method": method, "treatment": METHOD_NAMES.get(method, "Minimal baseline" if arm == "baseline" else "Preprocessed"),
                "cv_f1": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores)),
                "fold_scores": fold_scores, "removed": int(mask.sum()),
                "training_rows": int((~mask).sum()),
            })
    # Selection is locked before any held-out test predictions are computed.
    selected = max(results, key=lambda row: row["cv_f1"])
    for row in results:
        pipeline = fitted[row["id"]][0]
        probabilities = pipeline.predict_proba(Xtest)
        row.update(metrics(ytest, pipeline.predict(Xtest), probabilities, classes))
    pipeline, mask, scores = fitted[selected["id"]]
    selected_predictions = pipeline.predict(Xtest)
    probabilities = pipeline.predict_proba(Xtest)
    baseline = next(row for row in results if row["model_key"] == selected["model_key"] and row["arm"] == "baseline")
    selected_preprocessed = next(row for row in results if row["model_key"] == selected["model_key"] and row["arm"] == "preprocessed")
    outlier_candidates = [row for row in results if row["arm"] == "outlier"]
    best_outlier = max(outlier_candidates, key=lambda row: row["cv_f1"]) if outlier_candidates else None
    best_outlier_baseline = (
        next(row for row in results if row["model_key"] == best_outlier["model_key"] and row["arm"] == "baseline")
        if best_outlier else None
    )
    best_outlier_preprocessed = (
        next(row for row in results if row["model_key"] == best_outlier["model_key"] and row["arm"] == "preprocessed")
        if best_outlier else None
    )
    transformed = preprocessor(Xtrain, config).fit_transform(Xtrain)
    coordinates = PCA(n_components=min(2, *transformed.shape), random_state=config.seed).fit_transform(transformed)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates, np.zeros(len(coordinates))])
    scatter = [
        {"x": float(coordinates[i, 0]), "y": float(coordinates[i, 1]),
         "class": str(ytrain.iloc[i]), "outlier": bool(mask[i]), "row": int(source_rows[train_ids[i]]),
         "score": float(scores[i])}
        for i in range(len(Xtrain))
    ]
    numeric = Xtrain.select_dtypes(include="number")
    correlations = numeric.corr().fillna(0).round(3)
    correlations = correlations.iloc[:12, :12]
    def summarize_quantiles(removal_mask):
        summaries = []
        for col in numeric.columns[:8]:
            before, after = numeric[col].dropna(), numeric.loc[~removal_mask, col].dropna()
            summaries.append({
                "feature": col,
                "before": np.quantile(before, [0, .25, .5, .75, 1]).tolist() if len(before) else None,
                "after": np.quantile(after, [0, .25, .5, .75, 1]).tolist() if len(after) else None,
            })
        return summaries

    quantiles = summarize_quantiles(mask)
    quantiles_by_method = {"none": summarize_quantiles(np.zeros(len(Xtrain), dtype=bool))}
    roc = None
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(np.asarray(ytest) == classes[1], probabilities[:, 1])
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "positive_class": classes[1]}
    outlier_views = {}
    for row in results:
        if row["method"] and row["method"] not in outlier_views:
            _, view_mask, view_scores = fitted[row["id"]]
            outlier_views[row["method"]] = {
                "name": row["treatment"], "mask": view_mask.tolist(), "scores": view_scores.tolist(),
                "removed": int(view_mask.sum()),
                "classes": [
                    {"class": str(label), "total": int((ytrain == label).sum()),
                     "removed": int(((ytrain == label) & view_mask).sum())}
                    for label in classes
                ],
            }
            quantiles_by_method[row["method"]] = summarize_quantiles(view_mask)
    report = {
        "dataset": dataset.name, "notes": dataset.notes, "profile": profile,
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "Flask", "joblib")
        }},
        "config": asdict(config), "classes": classes, "results": results,
        "selected_id": selected["id"], "baseline_id": baseline["id"],
        "preprocessed_id": selected_preprocessed["id"],
        "best_outlier_id": best_outlier["id"] if best_outlier else None,
        "accuracy_delta": selected["accuracy"] - baseline["accuracy"],
        "outlier_accuracy_delta": best_outlier["accuracy"] - best_outlier_baseline["accuracy"] if best_outlier else None,
        "outlier_only_accuracy_delta": best_outlier["accuracy"] - best_outlier_preprocessed["accuracy"] if best_outlier else None,
        "split": {
            "train": len(train_ids), "test": len(test_ids),
            "train_ids": train_ids.tolist(), "test_ids": test_ids.tolist(),
            "train_source_rows": source_rows[train_ids].tolist(), "test_source_rows": source_rows[test_ids].tolist(),
        },
        "scatter": scatter, "outlier_views": outlier_views,
        "sample_input": Xtest.head(1).astype(object).where(Xtest.head(1).notna(), None).to_dict("records"),
        "correlation": {"columns": list(correlations.columns), "values": correlations.values.tolist()},
        "quantiles": quantiles, "quantiles_by_method": quantiles_by_method,
        "roc": roc, "rejected": rejected, "warnings": sorted(warnings_seen),
        "seconds": round(perf_counter() - started, 2),
        "protocol": [
            "Blank-label rows are removed; exact feature duplicates are removed before splitting. Conflicting duplicate labels are rejected.",
            "Export source_row values are zero-based original data-record positions, excluding the CSV header. Split train_ids/test_ids index the cleaned dataset.",
            "One seeded stratified holdout is shared by every model and treatment. The test set is never filtered.",
            "Within every CV fold, imputation, encoding, scaling, PCA, and outlier detection fit on that fold's training subset only.",
            "Outlier detectors use numeric features without labels. Only training rows are removed; preprocessing then refits on retained training rows.",
            "The winner is selected by mean stratified training-CV macro-F1 before test predictions. Test metrics for all arms are exploratory, not a tuning signal.",
            "The baseline still imputes and encodes so all classifiers can run. Full preprocessing adds scaling and optional 95%-variance PCA.",
            "Outlier removal can reduce accuracy and remove valid minority observations. Negative and zero gains are retained honestly.",
        ],
    }
    predictions = Xtest.add_prefix("feature__").copy()
    predictions.insert(0, "source_row", source_rows[test_ids])
    predictions["actual_class"] = np.asarray(ytest)
    predictions["predicted_class"] = selected_predictions
    predictions["correct"] = np.asarray(ytest) == selected_predictions
    for i, label in enumerate(classes):
        predictions[f"probability_{label}"] = probabilities[:, i]
    audit = Xtrain.add_prefix("feature__").copy()
    audit.insert(0, "source_row", source_rows[train_ids])
    audit["actual_class"] = np.asarray(ytrain)
    audit["is_outlier"] = mask
    audit["outlier_score"] = scores
    retained = Xtrain.loc[~mask].copy()
    retained_transformed = pipeline.named_steps["preprocess"].transform(retained)
    processor = pipeline.named_steps["preprocess"]
    if config.pca and selected["arm"] != "baseline":
        feature_names = [f"PC{i + 1}" for i in range(retained_transformed.shape[1])]
    else:
        feature_names = processor.named_steps["columns"].get_feature_names_out().tolist()
    cleaned = pd.DataFrame(retained_transformed, columns=[f"feature__{name}" for name in feature_names])
    cleaned.insert(0, "source_row", source_rows[train_ids[~mask]])
    cleaned["class"] = np.asarray(ytrain.loc[~mask])
    return report, pipeline, predictions, audit, cleaned
