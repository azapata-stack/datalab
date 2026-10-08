"""Experimento A/B con respuesta continua."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datalab.analysis import compare_groups, csv_bytes
from datalab.ui import cached_ab, intro, seed_control


def render():
    intro(
        "⚖️ Inferencia estadística A/B",
        "Compara dos grupos independientes con una respuesta continua.",
    )
    seed = seed_control("ab_seed")
    n = st.sidebar.slider("Tamaño por grupo", 20, 5000, 200, 20)
    effect = st.sidebar.slider("Efecto verdadero (Tratamiento − Control)", -10.0, 10.0, 2.0, 0.5)
    sigma = st.sidebar.slider("Desviación estándar poblacional", 1.0, 30.0, 10.0, 1.0)
    confidence = st.sidebar.selectbox("Nivel de confianza", [0.90, 0.95, 0.99], index=1)
    frame = cached_ab(n, effect, sigma, seed)
    result = compare_groups(frame, confidence)
    alpha = 1 - confidence
    left, middle, right = st.columns(3)
    left.metric("Diferencia observada", f"{result['difference']:.3f}")
    middle.metric(
        f"IC del {confidence:.0%}", f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
    )
    right.metric("Hedges g", f"{result['hedges_g']:.3f}")
    st.write(
        f"**Prueba t de Welch bilateral:** t = {result['t_statistic']:.3f}, "
        f"p = {result['p_value']:.4g}, α = {alpha:.2f}. "
        "Hipótesis nula: la diferencia de medias poblacionales es cero."
    )
    if result["p_value"] < alpha:
        st.info("Esta muestra aporta evidencia contra la hipótesis nula al nivel seleccionado.")
    else:
        st.info("Esta muestra no aporta evidencia suficiente para rechazar la hipótesis nula.")
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(
            frame, x="resultado", color="grupo", nbins=40, barmode="overlay", opacity=0.65
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.box(frame, x="grupo", y="resultado", color="grupo"), use_container_width=True
    )
    interval = go.Figure(
        go.Scatter(
            x=[result["difference"]],
            y=["Diferencia observada"],
            mode="markers",
            marker={"size": 14, "color": "#0F766E"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [result["ci_upper"] - result["difference"]],
                "arrayminus": [result["difference"] - result["ci_lower"]],
            },
        )
    )
    interval.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Sin efecto")
    interval.add_vline(x=effect, line_color="orange", annotation_text="Efecto verdadero")
    interval.update_layout(height=260, xaxis_title="Tratamiento − Control", showlegend=False)
    st.plotly_chart(interval, use_container_width=True)
    means = frame.groupby("grupo")["resultado"].agg(["count", "mean", "std"]).reset_index()
    st.dataframe(means, hide_index=True, use_container_width=True)
    with st.expander("Supuestos e interpretación", expanded=True):
        st.markdown(
            "- La simulación asigna **n observaciones independientes por grupo**, con resultados "
            "normales y media de control 50. No representa tasas de conversión.\n"
            "- Welch permite varianzas desiguales. Su intervalo usa la aproximación "
            "de Welch–Satterthwaite; SciPy calcula la prueba y statsmodels el intervalo.\n"
            "- Un IC del 95% es un procedimiento que cubriría el efecto verdadero en cerca "
            "del 95% de las repeticiones; no asigna probabilidad a este intervalo concreto.\n"
            "- Hedges g estandariza la diferencia con la desviación agrupada y corrige el "
            "sesgo de muestras pequeñas. Su relevancia depende del contexto.\n"
            "- p no es la probabilidad de que la hipótesis nula sea verdadera. No rechazarla "
            "no demuestra equivalencia. Un efecto significativo puede ser poco relevante."
        )
    st.warning(
        "Cambiar repetidamente la semilla o el tamaño hasta obtener p < α aumenta los falsos "
        "positivos. Define el diseño antes de observar los resultados; esta demo no corrige "
        "pruebas múltiples ni monitoreo secuencial."
    )
    with st.expander("Resultados y datos exportables"):
        st.dataframe(pd.DataFrame([result]), hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar experimento (CSV)", csv_bytes(frame), "experimento_ab.csv", "text/csv"
        )
