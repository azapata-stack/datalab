"""Modelos con separación previa y preprocesamiento ajustado solo en entrenamiento."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    predictions: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray
    confusion: np.ndarray | None = None


def evaluate_predictions(task: str, actual, prediction, probability=None) -> dict[str, float]:
    if task == "regresion":
        return {
            "MAE": float(mean_absolute_error(actual, prediction)),
            "RMSE": float(root_mean_squared_error(actual, prediction)),
            "R²": float(r2_score(actual, prediction)),
        }
    return {
        "Exactitud": float(accuracy_score(actual, prediction)),
        "Exactitud balanceada": float(balanced_accuracy_score(actual, prediction)),
        "F1": float(f1_score(actual, prediction, zero_division=0)),
        "ROC AUC": float(roc_auc_score(actual, probability)),
    }


def fit_predictive_model(
    features: pd.DataFrame,
    target: pd.Series,
    task: str,
    model: str = "Lineal",
    test_size: float = 0.25,
    seed: int = 42,
) -> ModelResult:
    if task not in {"regresion", "clasificacion"} or model not in {"Lineal", "Bosque"}:
        raise ValueError("Tarea o modelo desconocido.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target if task == "clasificacion" else None,
    )
    if task == "regresion":
        estimator = (
            Ridge(alpha=1.0)
            if model == "Lineal"
            else RandomForestRegressor(
                n_estimators=80, max_depth=10, min_samples_leaf=3, random_state=seed, n_jobs=2
            )
        )
        baseline = DummyRegressor(strategy="mean")
    else:
        estimator = (
            LogisticRegression(max_iter=1000, random_state=seed)
            if model == "Lineal"
            else RandomForestClassifier(
                n_estimators=80, max_depth=10, min_samples_leaf=3, random_state=seed, n_jobs=2
            )
        )
        baseline = DummyClassifier(strategy="prior")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    baseline.fit(x_train, y_train)
    baseline_prediction = baseline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1] if task == "clasificacion" else None
    baseline_probability = baseline.predict_proba(x_test)[:, 1] if task == "clasificacion" else None
    predictions = pd.DataFrame({"real": y_test, "prediccion": prediction})
    predictions["error"] = predictions["real"] - predictions["prediccion"]
    if probability is not None:
        predictions["probabilidad"] = probability
        predictions["acierto"] = predictions["real"] == predictions["prediccion"]
    return ModelResult(
        pipeline=pipeline,
        metrics=evaluate_predictions(task, y_test, prediction, probability),
        baseline_metrics=evaluate_predictions(
            task, y_test, baseline_prediction, baseline_probability
        ),
        predictions=predictions,
        train_indices=x_train.index.to_numpy(),
        test_indices=x_test.index.to_numpy(),
        confusion=confusion_matrix(y_test, prediction, labels=[0, 1])
        if task == "clasificacion"
        else None,
    )
