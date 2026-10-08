import numpy as np
import pandas as pd
import pytest

from datalab.data import predictive_data
from datalab.models import fit_predictive_model


# Un escenario claramente separable comprueba aprendizaje, no un umbral de negocio
# arbitrario sobre clases superpuestas; las métricas se contrastan además por fórmula.
@pytest.mark.parametrize("task,noise", [("regresion", 10), ("clasificacion", 2.5)])
@pytest.mark.parametrize("model", ["Lineal", "Bosque"])
def test_models_train_and_score_held_out_data(task, noise, model):
    features, target = predictive_data(task, n=600, noise=noise)
    result = fit_predictive_model(features, target, task, model)
    assert len(result.train_indices) == 450
    assert len(result.test_indices) == 150
    assert not set(result.train_indices) & set(result.test_indices)
    assert set(result.train_indices) | set(result.test_indices) == set(features.index)
    assert set(result.predictions.index) == set(result.test_indices)
    assert all(np.isfinite(value) for value in result.metrics.values())
    if task == "regresion":
        assert result.metrics["RMSE"] < result.baseline_metrics["RMSE"]
        assert result.metrics["R²"] > 0.3
        assert result.confusion is None
    else:
        assert result.metrics["ROC AUC"] > 0.75
        assert result.confusion.sum() == len(result.test_indices)
        assert result.predictions["probabilidad"].between(0, 1).all()
        assert result.metrics["Exactitud balanceada"] > 0.65
        train_rate = target.loc[result.train_indices].mean()
        test_rate = target.loc[result.test_indices].mean()
        assert abs(train_rate - test_rate) < 0.02


@pytest.mark.parametrize("model", ["Lineal", "Bosque"])
def test_classification_metrics_against_predictions_on_overlapping_classes(model):
    features, target = predictive_data("clasificacion", n=600, noise=1.5)
    result = fit_predictive_model(features, target, "clasificacion", model)
    actual = result.predictions["real"].to_numpy()
    predicted = result.predictions["prediccion"].to_numpy()
    probability = result.predictions["probabilidad"].to_numpy()
    assert result.metrics["Exactitud"] == pytest.approx((actual == predicted).mean())
    positive = probability[actual == 1]
    negative = probability[actual == 0]
    # AUC = probabilidad de ordenar un positivo por encima de un negativo (empates 1/2).
    expected_auc = (positive[:, None] > negative).mean()
    expected_auc += 0.5 * (positive[:, None] == negative).mean()
    assert result.metrics["ROC AUC"] == pytest.approx(expected_auc)
    assert result.metrics["ROC AUC"] > result.baseline_metrics["ROC AUC"]
    tn, fp, fn, tp = result.confusion.ravel()
    assert result.metrics["F1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert result.metrics["Exactitud balanceada"] == pytest.approx(
        (tp / (tp + fn) + tn / (tn + fp)) / 2
    )


def test_preprocessing_learns_only_training_even_when_test_is_poisoned():
    features, target = predictive_data("regresion", n=500, noise=10)
    first = fit_predictive_model(features, target, "regresion")
    poisoned = features.copy()
    poisoned.loc[first.test_indices, :] = 1_000_000
    second = fit_predictive_model(poisoned, target, "regresion")
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    expected_medians = features.loc[first.train_indices].median().to_numpy()
    np.testing.assert_allclose(second.pipeline.named_steps["imputer"].statistics_, expected_medians)
    imputed_train = features.loc[first.train_indices].fillna(
        features.loc[first.train_indices].median()
    )
    np.testing.assert_allclose(
        second.pipeline.named_steps["scaler"].mean_, imputed_train.mean().to_numpy()
    )
    np.testing.assert_allclose(
        first.pipeline.named_steps["model"].coef_, second.pipeline.named_steps["model"].coef_
    )
    # No cambia el modelo al alterar exclusivamente los datos reservados.
    np.testing.assert_allclose(
        first.pipeline.predict(features.loc[first.train_indices]),
        second.pipeline.predict(features.loc[first.train_indices]),
    )


def test_model_run_reproducible():
    features, target = predictive_data("clasificacion", n=500, noise=1)
    one = fit_predictive_model(features, target, "clasificacion", "Bosque", seed=9)
    two = fit_predictive_model(features, target, "clasificacion", "Bosque", seed=9)
    assert one.metrics == two.metrics
    pd.testing.assert_frame_equal(one.predictions, two.predictions)
