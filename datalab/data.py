"""Generadores deterministas; no dependen de Streamlit ni de servicios externos."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

REGIONS = ("Norte", "Centro", "Sur")
CHANNELS = ("Web", "Tienda", "App")
SEGMENTS = ("Personas", "Pymes", "Empresas")


def customer_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Clientes ficticios con dependencia entre variables, faltantes y atípicos."""
    if n < 1000:
        raise ValueError("La demostración requiere al menos 1.000 observaciones.")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 13, n), 18, 85).round().astype(int)
    tenure = np.clip(rng.gamma(2, 14, n), 1, 120).round().astype(int)
    orders = rng.poisson(2 + tenure / 12, n)
    spend = np.round((orders + 1) * rng.lognormal(3.7, 0.45, n), 2)
    outliers = rng.choice(n, max(1, n // 100), replace=False)
    spend[outliers] *= 7
    satisfaction = np.clip(7.3 + tenure / 100 - spend / 8000 + rng.normal(0, 1.4, n), 1, 10)
    churn_probability = 1 / (1 + np.exp(-(-0.6 - tenure / 45 + (7 - satisfaction) * 0.5)))
    frame = pd.DataFrame(
        {
            "edad": age,
            "antiguedad_meses": tenure,
            "pedidos": orders,
            "gasto_mensual": spend,
            "satisfaccion": satisfaction.round(1),
            "region": rng.choice(REGIONS, n, p=[0.25, 0.5, 0.25]),
            "canal": rng.choice(CHANNELS, n, p=[0.45, 0.3, 0.25]),
            "abandono": rng.binomial(1, churn_probability),
        }
    )
    for column, rate in [("gasto_mensual", 0.04), ("satisfaccion", 0.08), ("canal", 0.03)]:
        frame.loc[rng.random(n) < rate, column] = np.nan
    return frame


def ab_data(n: int, effect: float, sigma: float, seed: int = 42) -> pd.DataFrame:
    """Resultados continuos de dos grupos independientes de igual tamaño."""
    if n < 2 or sigma <= 0:
        raise ValueError("Se requiere n >= 2 y una desviación estándar positiva.")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "grupo": np.repeat(["Control", "Tratamiento"], n),
            "resultado": np.concatenate(
                [rng.normal(50, sigma, n), rng.normal(50 + effect, sigma, n)]
            ),
        }
    )


def predictive_data(task: str, n: int = 1500, noise: float = 15, seed: int = 42):
    """Predictores con faltantes; ruido de regresión o separación de clases."""
    if task == "regresion":
        x, y = make_regression(
            n_samples=n, n_features=6, n_informative=4, noise=noise, random_state=seed
        )
    elif task == "clasificacion":
        x, y = make_classification(
            n_samples=n,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            weights=[0.65, 0.35],
            class_sep=noise,
            flip_y=0.04,
            random_state=seed,
        )
    else:
        raise ValueError("Tarea desconocida.")
    frame = pd.DataFrame(x, columns=[f"variable_{i + 1}" for i in range(6)])
    rng = np.random.default_rng(seed + 1)
    frame = frame.mask(rng.random(frame.shape) < 0.04)
    return frame, pd.Series(y, name="objetivo")


def organization_data(seed: int = 42) -> pd.DataFrame:
    """Ventas diarias de NovaData; un registro por fecha, región y segmento."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    rows = []
    for day_index, date in enumerate(dates):
        season = 1 + 0.15 * np.sin(2 * np.pi * day_index / 365)
        weekday = 0.75 if date.dayofweek >= 5 else 1.1
        trend = 1 + day_index / 1000
        for region_index, region in enumerate(REGIONS):
            for segment_index, segment in enumerate(SEGMENTS):
                base = (28 + region_index * 10 + segment_index * 8) * season * weekday * trend
                orders = int(rng.poisson(base))
                ticket = (45 + segment_index * 35) * rng.lognormal(0, 0.1)
                revenue = round(orders * ticket, 2)
                costs = round(revenue * rng.uniform(0.55, 0.8), 2)
                target = round(base * (45 + segment_index * 35) * 1.05, 2)
                rows.append((date, region, segment, orders, revenue, costs, target))
    return pd.DataFrame(
        rows,
        columns=["fecha", "region", "segmento", "pedidos", "ingresos", "costos", "meta_ingresos"],
    )
