"""Regresión y clasificación sin fuga entre entrenamiento y prueba."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from datalab.analysis import csv_bytes
from datalab.ui import cached_model, cached_predictive_data, intro, seed_control


def render():
    intro(
        "🎯 Modelos predictivos",
        "Entrena un modelo y examina su desempeño sobre observaciones reservadas.",
    )
    seed = seed_control("ml_seed")
    task_label = st.sidebar.radio("Tarea", ["Regresión", "Clasificación"])
    task = "regresion" if task_label == "Regresión" else "clasificacion"
    model = st.sidebar.selectbox("Modelo", ["Lineal", "Bosque"])
    n = st.sidebar.slider("Observaciones del modelo", 500, 5000, 1500, 100)
    test_size = st.sidebar.slider("Proporción de prueba", 0.15, 0.40, 0.25, 0.05)
    if task == "regresion":
        difficulty = st.sidebar.slider(
            "Ruido de regresión (desviación estándar)", 1.0, 60.0, 15.0, 1.0
        )
    else:
        difficulty = st.sidebar.slider("Separación de clases", 0.3, 2.5, 1.0, 0.1)
    features, target = cached_predictive_data(task, n, difficulty, seed)
    result = cached_model(features, target, task, model, test_size, seed)
    st.caption(
        f"Entrenamiento: {len(result.train_indices):,} · Prueba: {len(result.test_indices):,} · "
        "6 predictores numéricos · aproximadamente 4% de celdas faltantes"
    )
    for column, (label, value) in zip(
        st.columns(len(result.metrics)), result.metrics.items(), strict=True
    ):
        column.metric(label, f"{value:.3f}")
    scores = pd.DataFrame({"Modelo": result.metrics, "Referencia simple": result.baseline_metrics})
    st.dataframe(
        scores.reset_index(names="Métrica"),
        hide_index=True,
        column_config={
            "Métrica": st.column_config.TextColumn(width="medium"),
            "Modelo": st.column_config.NumberColumn(format="%.3f"),
            "Referencia simple": st.column_config.NumberColumn(format="%.3f"),
        },
        use_container_width=True,
    )
    st.caption(
        "Referencia: media de entrenamiento para regresión; clase mayoritaria y proporción "
        "de entrenamiento para clasificación. Todas las métricas se calculan en prueba."
    )
    predictions = result.predictions
    if task == "regresion":
        left, right = st.columns(2)
        chart = px.scatter(predictions, x="real", y="prediccion", opacity=0.65)
        lower = float(predictions[["real", "prediccion"]].min().min())
        upper = float(predictions[["real", "prediccion"]].max().max())
        chart.add_trace(
            go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Predicción perfecta")
        )
        left.plotly_chart(chart, use_container_width=True)
        residuals = px.scatter(predictions, x="prediccion", y="error", opacity=0.65)
        residuals.add_hline(y=0, line_dash="dash")
        right.plotly_chart(residuals, use_container_width=True)
        st.plotly_chart(px.histogram(predictions, x="error", nbins=35), use_container_width=True)
        worst = predictions.assign(error_absoluto=predictions["error"].abs()).nlargest(
            20, "error_absoluto"
        )
        st.caption(
            "Error = real − predicción. Patrones en residuos pueden señalar problemas del modelo."
        )
    else:
        left, right = st.columns(2)
        left.plotly_chart(
            px.imshow(
                result.confusion,
                x=["Predicho 0", "Predicho 1"],
                y=["Real 0", "Real 1"],
                text_auto=True,
                color_continuous_scale="Teal",
            ),
            use_container_width=True,
        )
        fpr, tpr, _ = roc_curve(predictions["real"], predictions["probabilidad"])
        curve = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="Modelo"))
        curve.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"dash": "dash"})
        )
        curve.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad")
        right.plotly_chart(curve, use_container_width=True)
        chart_data = predictions.assign(clase_real=predictions["real"].astype(str))
        st.plotly_chart(
            px.histogram(
                chart_data, x="probabilidad", color="clase_real", nbins=30, barmode="overlay"
            ),
            use_container_width=True,
        )
        worst = predictions.loc[~predictions["acierto"]].copy()
        worst["confianza_error"] = worst["probabilidad"].where(
            worst["real"] == 0, 1 - worst["probabilidad"]
        )
        worst = worst.nlargest(20, "confianza_error")
        st.caption(
            "Clase positiva = 1; umbral = 0,5. La exactitud balanceada promedia el recall "
            "de las dos clases; ROC AUC evalúa ordenamiento, no calibración."
        )
    st.subheader("Errores de prueba para inspeccionar")
    st.dataframe(worst.join(features), use_container_width=True)
    st.download_button(
        "Descargar predicciones de prueba (CSV)",
        csv_bytes(predictions.reset_index(names="observacion")),
        "predicciones.csv",
        "text/csv",
    )
    with st.expander("Cómo se evita la fuga de información", expanded=True):
        st.markdown(
            "1. Se reserva la muestra de prueba antes de aprender cualquier transformación.\n"
            "2. Una `Pipeline` ajusta la mediana de imputación y `StandardScaler` "
            "**solo en entrenamiento**.\n"
            "3. El modelo recibe esas mismas transformaciones al predecir los datos de prueba.\n"
            "4. La clasificación usa partición estratificada; los bosques limitan su profundidad "
            "y usan dos hilos para respetar los recursos del entorno."
        )
    st.warning(
        "La muestra de prueba permanece fija con la misma semilla. Si eliges el mejor modelo "
        "mirando repetidamente estas métricas, la evaluación deja de ser independiente. "
        "En trabajo real, selecciona con validación cruzada dentro del entrenamiento y "
        "reserva un conjunto final adicional. Los datos sintéticos no validan generalización real."
    )
