"""Punto de entrada para Streamlit y Streamlit Community Cloud."""

import streamlit as st

from datalab.pages import dashboard, eda, home, inference, predictive

st.set_page_config(page_title="DataLab", page_icon="🧪", layout="wide")

navigation = st.navigation(
    [
        st.Page(home.render, title="Inicio", icon="🧪", default=True),
        st.Page(eda.render, title="Exploración de datos", icon="🔎", url_path="exploracion"),
        st.Page(inference.render, title="Inferencia A/B", icon="⚖️", url_path="inferencia"),
        st.Page(predictive.render, title="Modelos predictivos", icon="🎯", url_path="modelos"),
        st.Page(dashboard.render, title="Dashboard", icon="📊", url_path="dashboard"),
    ]
)
with st.sidebar:
    st.caption("DataLab · Datos sintéticos · Sin credenciales")
navigation.run()
