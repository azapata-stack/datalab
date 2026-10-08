"""Transformaciones y estadísticas reutilizables, sin elementos de interfaz."""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


def filter_customers(
    frame: pd.DataFrame, regions: list[str], channels: list[str], ages: tuple[int, int]
) -> pd.DataFrame:
    channel = frame["canal"].fillna("Sin dato")
    mask = frame["region"].isin(regions) & channel.isin(channels) & frame["edad"].between(*ages)
    return frame.loc[mask].copy()


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    return pd.DataFrame(
        {"faltantes": counts, "porcentaje": counts / max(len(frame), 1) * 100}
    ).rename_axis("variable")


def iqr_outliers(series: pd.Series, factor: float = 1.5) -> tuple[pd.Series, float, float]:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index), float("nan"), float("nan")
    q1, q3 = clean.quantile([0.25, 0.75])
    lower, upper = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    return ((series < lower) | (series > upper)).fillna(False), float(lower), float(upper)


def compare_groups(frame: pd.DataFrame, confidence: float = 0.95) -> dict[str, float]:
    """Welch bilateral; diferencia y CI definidos como Tratamiento − Control."""
    control = frame.loc[frame["grupo"] == "Control", "resultado"].dropna().to_numpy()
    treatment = frame.loc[frame["grupo"] == "Tratamiento", "resultado"].dropna().to_numpy()
    if min(len(control), len(treatment)) < 2 or not 0 < confidence < 1:
        raise ValueError("Se requieren dos observaciones por grupo y confianza entre 0 y 1.")
    variance_c, variance_t = control.var(ddof=1), treatment.var(ddof=1)
    if variance_c + variance_t == 0:
        raise ValueError("La comparación requiere variabilidad en los resultados.")
    comparison = CompareMeans(DescrStatsW(treatment), DescrStatsW(control))
    lower, upper = comparison.tconfint_diff(alpha=1 - confidence, usevar="unequal")
    test = stats.ttest_ind(treatment, control, equal_var=False)
    degrees = len(control) + len(treatment) - 2
    pooled = np.sqrt(
        ((len(control) - 1) * variance_c + (len(treatment) - 1) * variance_t) / degrees
    )
    difference = treatment.mean() - control.mean()
    cohen_d = difference / pooled
    return {
        "mean_control": float(control.mean()),
        "mean_treatment": float(treatment.mean()),
        "difference": float(difference),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "cohen_d": float(cohen_d),
        "hedges_g": float(cohen_d * (1 - 3 / (4 * degrees - 1))),
    }


def filter_organization(
    frame: pd.DataFrame,
    start,
    end,
    regions: list[str],
    segments: list[str],
) -> pd.DataFrame:
    mask = (
        frame["fecha"].between(pd.Timestamp(start), pd.Timestamp(end))
        & frame["region"].isin(regions)
        & frame["segmento"].isin(segments)
    )
    return frame.loc[mask].copy()


def dashboard_summary(frame: pd.DataFrame, frequency: str = "Semana"):
    """SQL sobre datos filtrados; ratios calculados desde sumas, sin promediar ratios."""
    periods = {"Día": "day", "Semana": "week", "Mes": "month"}
    if frequency not in periods:
        raise ValueError("Frecuencia no soportada.")
    with duckdb.connect(":memory:") as connection:
        connection.register("ventas", frame)
        totals = (
            connection.execute(
                """SELECT COALESCE(SUM(pedidos), 0) AS pedidos,
                      COALESCE(SUM(ingresos), 0) AS ingresos,
                      COALESCE(SUM(costos), 0) AS costos,
                      COALESCE(SUM(meta_ingresos), 0) AS meta_ingresos FROM ventas"""
            )
            .df()
            .iloc[0]
            .to_dict()
        )
        series = connection.execute(
            """SELECT date_trunc(?, fecha) AS periodo, SUM(pedidos) AS pedidos,
                      SUM(ingresos) AS ingresos, SUM(costos) AS costos,
                      SUM(meta_ingresos) AS meta_ingresos
               FROM ventas GROUP BY 1 ORDER BY 1""",
            [periods[frequency]],
        ).df()
    totals["beneficio"] = totals["ingresos"] - totals["costos"]
    totals["margen"] = (
        totals["beneficio"] / totals["ingresos"] if totals["ingresos"] else float("nan")
    )
    totals["ticket_medio"] = (
        totals["ingresos"] / totals["pedidos"] if totals["pedidos"] else float("nan")
    )
    totals["cumplimiento"] = (
        totals["ingresos"] / totals["meta_ingresos"] if totals["meta_ingresos"] else float("nan")
    )
    return totals, series


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 con BOM para exportación compatible con planillas."""
    return frame.to_csv(index=False).encode("utf-8-sig")
