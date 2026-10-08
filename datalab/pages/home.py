"""Portada y guía del laboratorio."""

import streamlit as st

from datalab.ui import intro


def render():
    intro(
        "🧪 DataLab",
        "Un laboratorio para **explorar, comparar, predecir y monitorear**. "
        "Cada demostración funciona con datos sintéticos, sin cuentas externas ni APIs.",
    )
    st.info(
        "Elige una demostración en la navegación lateral. Los controles recalculan los resultados."
    )
    cards = [
        (
            "🔎 Exploración de datos",
            "01 · Comprender",
            "Clientes ficticios: distribuciones, "
            "correlaciones, faltantes y valores atípicos. Filtra la muestra y descarga los datos.",
        ),
        (
            "⚖️ Inferencia A/B",
            "02 · Comparar",
            "Simula dos grupos, cambia el tamaño muestral, "
            "el efecto y la variabilidad. Interpreta diferencias, incertidumbre "
            "y tamaño del efecto.",
        ),
        (
            "🎯 Modelos predictivos",
            "03 · Predecir",
            "Regresión y clasificación con modelos "
            "lineales o bosques, una referencia simple y evaluación sobre datos de prueba.",
        ),
        (
            "📊 Dashboard",
            "04 · Monitorear",
            "Indicadores de NovaData, una organización ficticia. "
            "Explora ingresos, pedidos y margen por fecha, región y segmento; exporta tablas.",
        ),
    ]
    columns = st.columns(2)
    for index, (title, stage, description) in enumerate(cards):
        with columns[index % 2].container(border=True):
            st.caption(stage)
            st.subheader(title)
            st.write(description)
    st.subheader("Cómo experimentar")
    st.markdown(
        "1. Conserva la semilla para repetir un resultado.\n"
        "2. Cambia un parámetro por vez y observa su impacto.\n"
        "3. Lee los supuestos antes de interpretar una métrica.\n"
        "4. Descarga los datos para revisar los cálculos fuera de la aplicación."
    )
    st.warning(
        "Estas simulaciones ilustran métodos; sus resultados no describen personas ni "
        "organizaciones reales. La correlación no prueba causalidad y una buena métrica "
        "en datos sintéticos no garantiza desempeño en producción."
    )
