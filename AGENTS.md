# Mantenimiento de DataLab

- Trabaja en el checkout existente. Las tareas de Codex Cloud ya están aisladas:
  no crees worktrees salvo petición explícita.
- Python recomendado: 3.12. Instalación: `UV_CACHE_DIR=/workspace/.cache/uv uv sync --frozen`.
- Ejecuta desde la raíz: `uv run --frozen streamlit run app.py`.
  En Codex Cloud exporta `UV_CACHE_DIR=/workspace/.cache/uv` antes de todos los comandos uv.
- Mantén generación (`data.py`), procesamiento (`analysis.py`), modelos (`models.py`)
  y presentación (`pages/`, `ui.py`) separados. Las capas de cálculo no deben importar Streamlit.
- Todas las demos deben funcionar sin APIs externas, credenciales ni datos reales.
  Usa generadores locales `numpy.random.default_rng(seed)`; evita estado aleatorio global.
- Separa entrenamiento/prueba ANTES de aprender cualquier transformación. Imputación,
  escala, selección de variables y ajuste de modelos deben aprender solo de entrenamiento.
  Conserva referencias simples y métricas sobre la misma muestra de prueba.
- No borres atípicos silenciosamente, no interpretes p como probabilidad de la hipótesis,
  no promedies ratios sin considerar sus pesos y documenta supuestos junto al resultado.
- Conserva estados vacíos claros cuando filtros no seleccionen datos.
- Antes de entregar: `uv run --frozen pytest`, `uv run --frozen ruff check .` y
  `uv run --frozen ruff format --check .`. Los tests de AppTest no prueban rendering
  JavaScript, screenshots ni clics reales sobre descargas.
- Para cambios de dependencias, actualiza deliberadamente `uv.lock` y regenera
  `requirements.txt` con `uv export --frozen --no-dev --no-emit-project --no-hashes --no-header
  --output-file requirements.txt`. No actualices dependencias como efecto secundario de un test.
- No incluyas `.venv`, cachés, secretos, logs ni datos privados en Git.
- Community Cloud: entrada `app.py`, Python 3.12, sin secretos. No despliegues ni hagas
  push sin una instrucción del usuario; presenta primero el cambio para revisión.
