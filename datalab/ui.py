"""Componentes comunes y cachés de presentación."""

import streamlit as st

from datalab.data import ab_data, customer_data, organization_data, predictive_data
from datalab.models import fit_predictive_model

cached_customers = st.cache_data(show_spinner=False)(customer_data)
cached_ab = st.cache_data(show_spinner=False)(ab_data)
cached_organization = st.cache_data(show_spinner=False)(organization_data)
cached_predictive_data = st.cache_data(show_spinner=False)(predictive_data)
cached_model = st.cache_data(show_spinner="Entrenando con los datos de entrenamiento…")(
    fit_predictive_model
)


def seed_control(key: str) -> int:
    return int(
        st.sidebar.number_input(
            "Semilla aleatoria", min_value=0, max_value=1_000_000, value=42, step=1, key=key
        )
    )


def intro(title: str, description: str):
    st.title(title)
    st.markdown(description)
    st.caption("Datos simulados · resultados reproducibles · herramienta educativa")
