import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from datalab.analysis import (
    compare_groups,
    csv_bytes,
    dashboard_summary,
    filter_customers,
    filter_organization,
    iqr_outliers,
    missing_summary,
)
from datalab.data import ab_data, customer_data, organization_data, predictive_data


def test_customers_reproducible_realistic_and_missing():
    frame = customer_data()
    pd.testing.assert_frame_equal(frame, customer_data())
    assert len(frame) >= 1000
    assert len(frame.select_dtypes("number").columns) >= 4
    assert frame["region"].nunique() == 3
    assert frame.isna().sum().sum() > 0
    assert frame["edad"].between(18, 85).all()
    assert not frame.equals(customer_data(seed=7))
    with pytest.raises(ValueError):
        customer_data(999)


def test_customer_filters_missing_summary_and_no_mutation():
    frame = customer_data()
    original = frame.copy(deep=True)
    filtered = filter_customers(frame, ["Norte"], ["Web", "Sin dato"], (20, 50))
    assert not filtered.empty
    assert filtered["region"].eq("Norte").all()
    assert filtered["canal"].fillna("Sin dato").isin(["Web", "Sin dato"]).all()
    assert filtered["edad"].between(20, 50).all()
    pd.testing.assert_frame_equal(frame, original)
    summary = missing_summary(frame)
    assert summary.loc["satisfaccion", "faltantes"] == frame["satisfaccion"].isna().sum()
    assert summary.loc["satisfaccion", "porcentaje"] == pytest.approx(
        frame["satisfaccion"].isna().mean() * 100
    )
    assert filter_customers(frame, [], ["Web"], (18, 85)).empty


def test_iqr_known_outlier_and_missing():
    series = pd.Series([1, 2, 2, 3, 100, np.nan])
    mask, lower, upper = iqr_outliers(series)
    assert mask.tolist() == [False, False, False, False, True, False]
    assert lower == 0.5
    assert upper == 4.5
    empty_mask, lower, upper = iqr_outliers(pd.Series([np.nan]))
    assert not empty_mask.any()
    assert np.isnan(lower) and np.isnan(upper)


def test_ab_generation_reproducibility_and_true_effect():
    frame = ab_data(2000, effect=4, sigma=2)
    pd.testing.assert_frame_equal(frame, ab_data(2000, effect=4, sigma=2))
    result = compare_groups(frame)
    assert result["difference"] == pytest.approx(4, abs=0.2)
    assert result["ci_lower"] > 0
    assert result["p_value"] < 0.001
    assert result["hedges_g"] > 0
    assert result["hedges_g"] < result["cohen_d"]


def test_welch_against_independent_formula_unequal_groups():
    control = np.array([1, 2, 3, 4, 7], dtype=float)
    treatment = np.array([3, 6, 8, 9, 15, 17, 19], dtype=float)
    frame = pd.DataFrame(
        {
            "grupo": ["Control"] * len(control) + ["Tratamiento"] * len(treatment),
            "resultado": np.concatenate([control, treatment]),
        }
    )
    result = compare_groups(frame)
    c = control.var(ddof=1) / len(control)
    b = treatment.var(ddof=1) / len(treatment)
    se = np.sqrt(c + b)
    df = (c + b) ** 2 / (c**2 / (len(control) - 1) + b**2 / (len(treatment) - 1))
    difference = treatment.mean() - control.mean()
    critical = t.ppf(0.975, df)
    assert result["ci_lower"] == pytest.approx(difference - critical * se)
    assert result["ci_upper"] == pytest.approx(difference + critical * se)
    assert result["p_value"] == pytest.approx(2 * t.sf(abs(difference / se), df))
    wider = compare_groups(frame, confidence=0.99)
    assert wider["ci_lower"] < result["ci_lower"]
    assert wider["ci_upper"] > result["ci_upper"]


def test_negative_ab_effect_and_invalid_parameters():
    assert compare_groups(ab_data(1000, -5, 2))["difference"] < 0
    with pytest.raises(ValueError):
        ab_data(1, 2, 10)
    with pytest.raises(ValueError):
        ab_data(10, 2, 0)
    with pytest.raises(ValueError):
        compare_groups(ab_data(20, 2, 10), confidence=1)


@pytest.mark.parametrize("task,noise", [("regresion", 15), ("clasificacion", 1)])
def test_predictive_data_deterministic(task, noise):
    x, y = predictive_data(task, n=500, noise=noise)
    x2, y2 = predictive_data(task, n=500, noise=noise)
    pd.testing.assert_frame_equal(x, x2)
    pd.testing.assert_series_equal(y, y2)
    assert x.shape == (500, 6)
    assert x.isna().sum().sum() > 0


def test_organization_unique_grain_and_reproducibility():
    frame = organization_data()
    pd.testing.assert_frame_equal(frame, organization_data())
    assert len(frame) == 365 * 3 * 3
    assert not frame.duplicated(["fecha", "region", "segmento"]).any()
    assert frame["pedidos"].ge(0).all()
    assert frame["ingresos"].ge(frame["costos"]).all()


@pytest.mark.parametrize("frequency", ["Día", "Semana", "Mes"])
def test_dashboard_duckdb_totals_match_filtered_data(frequency):
    frame = organization_data()
    filtered = filter_organization(frame, "2025-03-01", "2025-03-31", ["Sur"], ["Pymes"])
    assert len(filtered) == 31
    totals, series = dashboard_summary(filtered, frequency)
    for column in ["ingresos", "costos", "pedidos", "meta_ingresos"]:
        assert totals[column] == pytest.approx(filtered[column].sum())
        assert series[column].sum() == pytest.approx(totals[column])
    assert totals["margen"] == pytest.approx(
        1 - filtered["costos"].sum() / filtered["ingresos"].sum()
    )
    assert totals["ticket_medio"] == pytest.approx(
        filtered["ingresos"].sum() / filtered["pedidos"].sum()
    )
    assert totals["cumplimiento"] == pytest.approx(
        filtered["ingresos"].sum() / filtered["meta_ingresos"].sum()
    )
    assert series["periodo"].is_monotonic_increasing


def test_empty_dashboard_and_csv_roundtrip():
    frame = organization_data()
    filtered = filter_organization(frame, "2025-01-01", "2025-01-31", [], ["Pymes"])
    totals, series = dashboard_summary(filtered)
    assert totals["ingresos"] == 0
    assert np.isnan(totals["margen"])
    assert series.empty
    exported = csv_bytes(frame.head(3))
    assert exported.startswith(b"\xef\xbb\xbf")
    import io

    restored = pd.read_csv(io.BytesIO(exported), parse_dates=["fecha"])
    pd.testing.assert_frame_equal(restored, frame.head(3))
