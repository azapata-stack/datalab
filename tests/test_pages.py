from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def page_test(module):
    return AppTest.from_string(
        f"import streamlit as st\n"
        f"from datalab.pages.{module} import render\n"
        f"st.set_page_config(layout='wide')\n"
        f"render()\n",
        default_timeout=30,
    )


def assert_no_errors(app):
    assert not app.exception, [exception.message for exception in app.exception]
    assert not app.error


def test_app_entry_and_home():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert_no_errors(app)
    assert app.title[0].value == "🧪 DataLab"
    assert len(app.subheader) >= 4


@pytest.mark.parametrize("module", ["eda", "inference", "predictive", "dashboard"])
def test_demo_pages_render_charts_and_tables(module):
    app = page_test(module).run()
    assert_no_errors(app)
    assert len(app.get("plotly_chart")) >= 2
    assert len(app.dataframe) >= 1
    assert len(app.metric) >= 3
    assert len(app.get("download_button")) >= 1


def test_eda_filters_and_empty_selection():
    app = page_test("eda").run()
    app.sidebar.multiselect[0].set_value(["Norte"]).run()
    assert_no_errors(app)
    filtered = int(app.metric[0].value.replace(",", ""))
    assert 0 < filtered < 2000
    app.sidebar.multiselect[0].set_value([]).run()
    assert_no_errors(app)
    assert app.metric[0].value == "0"
    assert app.info


def test_ab_controls_recompute_results():
    app = page_test("inference").run()
    before = app.metric[0].value
    effect_slider = next(s for s in app.sidebar.slider if s.label.startswith("Efecto verdadero"))
    effect_slider.set_value(-5.0).run()
    assert_no_errors(app)
    assert float(app.metric[0].value) < 0
    assert app.metric[0].value != before
    app.sidebar.selectbox[0].set_value(0.99).run()
    assert_no_errors(app)
    assert app.metric[1].label == "IC del 99%"


def test_predictive_classification_and_forest():
    app = page_test("predictive").run()
    assert app.metric[0].label == "MAE"
    app.sidebar.radio[0].set_value("Clasificación").run()
    assert_no_errors(app)
    assert app.metric[0].label == "Exactitud"
    assert len(app.metric) == 4
    app.sidebar.selectbox[0].set_value("Bosque").run()
    assert_no_errors(app)
    assert len(app.get("plotly_chart")) >= 3


def test_dashboard_filters_frequency_and_empty_selection():
    app = page_test("dashboard").run()
    original_income = float(app.metric[0].value.replace(",", ""))
    app.sidebar.multiselect[0].set_value(["Sur"]).run()
    assert_no_errors(app)
    assert 0 < float(app.metric[0].value.replace(",", "")) < original_income
    app.sidebar.selectbox[0].set_value("Mes").run()
    assert_no_errors(app)
    app.sidebar.multiselect[1].set_value([]).run()
    assert_no_errors(app)
    assert not app.metric
    assert app.info
