"""Demostración de análisis exploratorio."""

import plotly.express as px
import streamlit as st

from datalab.analysis import csv_bytes, filter_customers, iqr_outliers, missing_summary
from datalab.data import CHANNELS, REGIONS
from datalab.ui import cached_customers, intro, seed_control


def render():
    intro(
        "🔎 Exploración de datos",
        "Explora una cartera sintética de clientes y su calidad de datos.",
    )
    seed = seed_control("eda_seed")
    n = st.sidebar.slider("Observaciones", 1000, 10000, 2000, 500)
    frame = cached_customers(n, seed)
    regions = st.sidebar.multiselect("Regiones", REGIONS, default=list(REGIONS), key="eda_regions")
    channel_options = [*CHANNELS, "Sin dato"]
    channels = st.sidebar.multiselect("Canales", channel_options, default=channel_options)
    ages = st.sidebar.slider("Edad", 18, 85, (18, 85))
    filtered = filter_customers(frame, regions, channels, ages)
    columns = st.columns(3)
    columns[0].metric("Observaciones filtradas", f"{len(filtered):,}")
    columns[1].metric("Observaciones originales", f"{len(frame):,}")
    columns[2].metric("Variables", len(frame.columns))
    if filtered.empty:
        st.info("No hay observaciones para estos filtros. Amplía la selección.")
        return
    numeric = filtered.select_dtypes("number").columns.tolist()
    tabs = st.tabs(["Distribuciones", "Correlaciones", "Faltantes", "Atípicos", "Datos"])
    with tabs[0]:
        variable = st.selectbox("Variable numérica", numeric, index=numeric.index("gasto_mensual"))
        group = st.selectbox("Agrupar por", ["region", "canal"])
        chart_data = filtered.assign(canal=filtered["canal"].fillna("Sin dato"))
        left, right = st.columns(2)
        left.plotly_chart(
            px.histogram(chart_data, x=variable, color=group, nbins=40, barmode="overlay"),
            use_container_width=True,
        )
        right.plotly_chart(
            px.box(chart_data, x=group, y=variable, color=group), use_container_width=True
        )
        st.dataframe(filtered[numeric].describe().T, use_container_width=True)
        st.caption(
            "Los gráficos omiten los faltantes de la variable numérica; no se imputan datos."
        )
    with tabs[1]:
        method = st.radio("Correlación", ["Pearson", "Spearman"], horizontal=True)
        corr = filtered[numeric].corr(method=method.lower())
        st.plotly_chart(
            px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f"),
            use_container_width=True,
        )
        st.caption(
            "Pearson mide asociación lineal; Spearman, asociación monótona entre rangos. "
            "Cada par usa solo sus observaciones completas, por lo que puede tener distinto n. "
            "Una correlación indefinida aparece vacía."
        )
        st.warning(
            "Asociación no implica causalidad. Filtrar puede cambiar las relaciones observadas."
        )
    with tabs[2]:
        summary = missing_summary(filtered).reset_index()
        st.plotly_chart(
            px.bar(summary, x="variable", y="porcentaje", labels={"porcentaje": "% faltante"}),
            use_container_width=True,
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(
            "Los faltantes se introducen aleatoriamente para esta demo. En datos reales, "
            "el mecanismo de ausencia puede sesgar los resultados."
        )
    with tabs[3]:
        outlier_variable = st.selectbox(
            "Variable para detectar atípicos", numeric, index=numeric.index("gasto_mensual")
        )
        factor = st.slider("Multiplicador del rango intercuartílico (IQR)", 1.0, 3.0, 1.5, 0.1)
        mask, lower, upper = iqr_outliers(filtered[outlier_variable], factor)
        valid_count = filtered[outlier_variable].notna().sum()
        st.metric("Atípicos detectados", int(mask.sum()))
        if valid_count:
            st.write(
                f"Límites: {lower:,.2f} a {upper:,.2f}; calculados sobre {valid_count:,} valores."
            )
            st.plotly_chart(
                px.scatter(
                    filtered.assign(observacion=filtered.index, atipico=mask.astype(str)),
                    x="observacion",
                    y=outlier_variable,
                    color="atipico",
                ),
                use_container_width=True,
            )
            st.dataframe(filtered.loc[mask], use_container_width=True)
        else:
            st.info("No hay valores numéricos disponibles para detectar atípicos.")
        st.caption(
            "Regla: Q1 − k·IQR y Q3 + k·IQR. Un punto marcado no es necesariamente un error. "
            "La asimetría del gasto puede producir muchos puntos marcados. "
            "No se eliminan registros."
        )
    with tabs[4]:
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar datos filtrados (CSV)", csv_bytes(filtered), "clientes.csv", "text/csv"
        )
