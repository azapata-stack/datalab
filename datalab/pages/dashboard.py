"""Panel de indicadores sobre ventas sintéticas de NovaData."""

import plotly.express as px
import streamlit as st

from datalab.analysis import csv_bytes, dashboard_summary, filter_organization
from datalab.data import REGIONS, SEGMENTS
from datalab.ui import cached_organization, intro, seed_control


def render():
    intro(
        "📊 Dashboard de NovaData",
        "Una organización ficticia: ventas, costos y metas durante 2025.",
    )
    seed = seed_control("dashboard_seed")
    frame = cached_organization(seed)
    minimum, maximum = frame["fecha"].min().date(), frame["fecha"].max().date()
    dates = st.sidebar.date_input(
        "Período", (minimum, maximum), min_value=minimum, max_value=maximum
    )
    regions = st.sidebar.multiselect("Regiones", REGIONS, default=list(REGIONS), key="dash_regions")
    segments = st.sidebar.multiselect("Segmentos", SEGMENTS, default=list(SEGMENTS))
    frequency = st.sidebar.selectbox("Agrupación temporal", ["Día", "Semana", "Mes"], index=1)
    if len(dates) != 2:
        st.info("Selecciona la fecha inicial y final del período.")
        return
    filtered = filter_organization(frame, dates[0], dates[1], regions, segments)
    if filtered.empty:
        st.info("No hay datos para estos filtros. Amplía la selección.")
        return
    totals, series = dashboard_summary(filtered, frequency)
    columns = st.columns(4)
    columns[0].metric("Ingresos (UM)", f"{totals['ingresos']:,.0f}")
    columns[1].metric("Pedidos", f"{totals['pedidos']:,.0f}")
    columns[2].metric("Margen", f"{totals['margen']:.1%}")
    columns[3].metric("Cumplimiento de meta", f"{totals['cumplimiento']:.1%}")
    st.caption(
        f"Beneficio: {totals['beneficio']:,.0f} UM · "
        f"Ticket medio: {totals['ticket_medio']:,.2f} UM · "
        f"{len(filtered):,} registros diarios filtrados. UM = unidades monetarias ficticias."
    )
    trend = series.melt("periodo", ["ingresos", "meta_ingresos"], var_name="serie", value_name="UM")
    st.plotly_chart(
        px.line(trend, x="periodo", y="UM", color="serie", markers=True), use_container_width=True
    )
    left, right = st.columns(2)
    left.plotly_chart(px.bar(series, x="periodo", y="pedidos"), use_container_width=True)
    by_region = filtered.groupby("region", as_index=False)[["ingresos", "costos"]].sum()
    right.plotly_chart(
        px.bar(by_region, x="region", y=["ingresos", "costos"], barmode="group"),
        use_container_width=True,
    )
    segment_totals = filtered.groupby("segmento", as_index=False)[
        ["ingresos", "costos", "pedidos"]
    ].sum()
    segment_totals["margen"] = (
        segment_totals["ingresos"] - segment_totals["costos"]
    ) / segment_totals["ingresos"]
    st.subheader("Resumen por segmento")
    st.dataframe(segment_totals, hide_index=True, use_container_width=True)
    with st.expander("Definiciones de los indicadores", expanded=True):
        st.markdown(
            "- **Ingresos / pedidos / costos / meta**: sumas de los registros seleccionados.\n"
            "- **Beneficio**: ingresos menos costos simulados; no representa utilidad contable.\n"
            "- **Margen**: beneficio total / ingresos totales.\n"
            "- **Ticket medio**: ingresos totales / pedidos totales.\n"
            "- **Cumplimiento**: ingresos totales / meta total.\n"
            "- DuckDB agrega las observaciones después de aplicar todos los filtros. "
            "Los ratios se calculan desde sumas, evitando promediar porcentajes "
            "con pesos distintos."
        )
    st.caption(
        "Las semanas comienzan el lunes. El primer y último período pueden ser parciales; "
        "su fecha de etiqueta puede preceder al filtro. No hay ajustes por inflación, "
        "devoluciones ni impuestos. Las metas son ilustrativas, no pronósticos."
    )
    with st.expander("Tablas y exportación"):
        st.dataframe(series, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar serie temporal (CSV)", csv_bytes(series), "serie_temporal.csv", "text/csv"
        )
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar registros filtrados (CSV)",
            csv_bytes(filtered),
            "ventas_filtradas.csv",
            "text/csv",
        )
