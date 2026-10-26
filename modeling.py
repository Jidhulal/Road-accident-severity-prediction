"""Shared data preparation, training, evaluation, and plotting utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "accident_severity"
CLASS_NAMES = ("Slight", "Serious", "Fatal")


def generate_demo_dataset(rows: int = 1200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible synthetic dataset for demonstrating the application."""
    rng = np.random.default_rng(random_state)
    vehicle_count = rng.integers(1, 6, size=rows)
    casualties = np.minimum(rng.poisson(1.3, size=rows) + 1, 8)
    speed_limit = rng.choice([30, 40, 50, 60, 70, 80, 100], size=rows,
                             p=[0.10, 0.15, 0.20, 0.22, 0.18, 0.10, 0.05])
    hour = rng.integers(0, 24, size=rows)
    weather = rng.choice(["Clear", "Rain", "Fog", "Snow"], size=rows,
                         p=[0.68, 0.22, 0.07, 0.03])
    road_surface = rng.choice(["Dry", "Wet", "Icy"], size=rows,
                              p=[0.70, 0.25, 0.05])
    light = np.where((hour < 6) | (hour >= 20), "Dark", "Daylight")
    area = rng.choice(["Urban", "Rural"], size=rows, p=[0.65, 0.35])

    risk_score = (
        (casualties - 1) * 0.38
        + (vehicle_count - 1) * 0.12
        + speed_limit / 100 * 0.75
        + np.isin(weather, ["Rain", "Fog", "Snow"]) * 0.34
        + np.isin(road_surface, ["Wet", "Icy"]) * 0.42
        + (light == "Dark") * 0.28
        + (area == "Rural") * 0.15
        + rng.normal(0, 0.48, size=rows)
    )
    slight_cutoff, serious_cutoff = np.quantile(risk_score, [0.65, 0.90])
    severity = np.where(
        risk_score > serious_cutoff,
        "Fatal",
        np.where(risk_score > slight_cutoff, "Serious", "Slight"),
    )

    return pd.DataFrame(
        {
            "vehicle_count": vehicle_count,
            "casualties": casualties,
            "speed_limit_kmh": speed_limit,
            "hour": hour,
            "weather": weather,
            "road_surface": road_surface,
            "light_condition": light,
            "area_type": area,
            TARGET_COLUMN: severity,
        }
    )


def _prepare_data(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if target_column not in data.columns:
        raise ValueError("Choose a target column that exists in the uploaded dataset.")

    original_rows = len(data)
    duplicate_count = int(data.duplicated().sum())
    missing_counts = data.isna().sum().to_dict()
    clean = data.drop_duplicates().dropna(subset=[target_column]).copy()
    clean = clean.dropna(axis=1, how="all")
    if target_column not in clean.columns:
        raise ValueError("The target column contains no usable values.")

    y = clean[target_column].astype(str)
    X = clean.drop(columns=[target_column])
    if X.empty:
        raise ValueError("The dataset must include at least one feature column.")
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("The target must contain at least two classes.")
    if int(counts.min()) < 5:
        raise ValueError("Each target class needs at least 5 rows for stratified 5-fold cross-validation.")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    info = {
        "original_rows": original_rows,
        "duplicate_rows": duplicate_count,
        "missing_values": {key: int(value) for key, value in missing_counts.items()},
        "rows_used": len(clean),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "class_counts": y.value_counts().sort_index().to_dict(),
    }
    return X, y, info


def _build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers)


def _plot_png(fig: plt.Figure, output_dir: Path | None, filename: str) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    image = buffer.getvalue()
    if output_dir is not None:
        (output_dir / filename).write_bytes(image)
    plt.close(fig)
    return image


def run_experiment(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train three leakage-safe pipelines and evaluate the CV-selected model."""
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    X, y, data_info = _prepare_data(data, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = _build_preprocessor(
        data_info["numeric_columns"], data_info["categorical_columns"]
    )
    candidates = {
        "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict[str, dict[str, float]] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
        )
        cv_results[name] = {
            "mean_f1_macro": float(scores.mean()),
            "std_f1_macro": float(scores.std()),
        }
        pipelines[name] = pipeline

    selected_name = max(cv_results, key=lambda name: cv_results[name]["mean_f1_macro"])
    model = pipelines[selected_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            auc = roc_auc_score(
                y_test, probabilities, labels=classes, multi_class="ovr", average="macro"
            )
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": float(auc),
    }
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    class_names = [str(value) for value in classes]
    chart_images: dict[str, bytes] = {}

    class_counts = pd.Series(data_info["class_counts"]).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index,
                palette="Blues", legend=False, ax=ax)
    ax.set(title="Accident severity class distribution", xlabel="Severity", ylabel="Records")
    chart_images["class_distribution"] = _plot_png(fig, output_path, "class_distribution.png")

    numeric_data = X[data_info["numeric_columns"]].copy()
    if not numeric_data.empty:
        for column in numeric_data.columns:
            if numeric_data[column].nunique(dropna=True) <= 1:
                numeric_data = numeric_data.drop(columns=[column])
    fig, ax = plt.subplots(figsize=(8, 6))
    if numeric_data.shape[1] >= 2:
        sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0, annot=True,
                    fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap (numeric features)")
    else:
        ax.text(0.5, 0.5, "At least two varying numeric features are needed.",
                ha="center", va="center")
        ax.set_axis_off()
    chart_images["correlation_heatmap"] = _plot_png(fig, output_path, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title="Confusion matrix", xlabel="Predicted severity", ylabel="Actual severity")
    chart_images["confusion_matrix"] = _plot_png(fig, output_path, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    label_binarizer = LabelBinarizer().fit(classes)
    binary_test = label_binarizer.transform(y_test)
    if len(classes) == 2:
        binary_test = np.column_stack([1 - binary_test.ravel(), binary_test.ravel()])
    for index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binary_test[:, index], probabilities[:, index])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC {roc_auc_score(binary_test[:, index], probabilities[:, index]):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(title="One-vs-rest ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    chart_images["roc_curve"] = _plot_png(fig, output_path, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(cv_results)
    means = [cv_results[name]["mean_f1_macro"] for name in names]
    deviations = [cv_results[name]["std_f1_macro"] for name in names]
    ax.bar(names, means, yerr=deviations, capsize=5, color=["#5B8FF9", "#61DDAA", "#65789B"])
    ax.set(title="Five-fold training CV: macro F1", ylabel="Mean macro F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=12)
    chart_images["model_comparison"] = _plot_png(fig, output_path, "model_comparison.png")

    return {
        "model": model,
        "selected_model": selected_name,
        "metrics": metrics,
        "cv_results": cv_results,
        "class_names": class_names,
        "confusion_matrix": matrix,
        "data_info": data_info,
        "feature_columns": X.columns.tolist(),
        "chart_images": chart_images,
    }
