# 🧪 DataLab

Laboratorio educativo multipágina, íntegramente en Python, para explorar datos,
comparar grupos, evaluar modelos y monitorear indicadores. Las cuatro demostraciones
usan datos sintéticos reproducibles y funcionan sin APIs, cuentas ni credenciales externas.

## Ejecutar

Requisitos: Python 3.12 (también se admite 3.13) y [uv](https://docs.astral.sh/uv/).
En Codex Cloud estas herramientas ya están disponibles; no se necesita Docker.

```bash
cd /workspace/datalab
export UV_CACHE_DIR=/workspace/.cache/uv
uv sync --frozen
uv run --frozen streamlit run app.py
```

En otro entorno, entra en la carpeta del repositorio y usa una ubicación escribible
para la caché de uv, o conserva su ubicación predeterminada. `uv sync` crea `.venv`
con dependencias de aplicación y desarrollo. La versión de Python recomendada está
en `.python-version`; `uv.lock` fija las versiones resueltas. En una máquina sin Python
3.12, uv puede instalarlo desde sus fuentes oficiales con `uv python install 3.12`.

Streamlit escucha en el puerto 8501. Para cambiarlo:

```bash
uv run --frozen streamlit run app.py --server.port 8502
```

### Demostraciones

| Página | Funciones |
| --- | --- |
| Exploración | 1.000–10.000 clientes ficticios, filtros, histogramas, cajas, Pearson/Spearman, faltantes, regla IQR y CSV. |
| Inferencia A/B | Dos grupos con resultados continuos normales; n por grupo, efecto, variabilidad y confianza ajustables; Welch bilateral, IC, Cohen d y Hedges g. |
| Modelos predictivos | Regresión Ridge o bosque; clasificación logística o bosque; partición reservada, imputación y escala en Pipeline, referencia simple, métricas, ROC, residuos y errores exportables. |
| Dashboard | NovaData: ventas de 2025 por día, región y segmento; filtros, agregaciones SQL con DuckDB, ingresos, pedidos, margen, metas y CSV. |

La semilla de cada página controla la generación. No se leen archivos externos ni se
persisten datos de usuarios. Las cachés de Streamlit solo conservan resultados sintéticos.

### Supuestos

- La correlación es descriptiva y usa observaciones completas por par. Los faltantes
  son aleatorios en la simulación; en un caso real pueden ser informativos.
- Los atípicos se señalan con la regla IQR, sin borrarlos. Un gasto alto no prueba un error.
- A/B supone independencia y una respuesta continua, no una proporción de conversión.
  Welch permite varianzas desiguales. El valor p no es una probabilidad de la hipótesis
  nula; un resultado no significativo no demuestra equivalencia. Cambiar semillas hasta
  obtener significación introduce selección. No hay corrección por pruebas múltiples.
- Los modelos reservan prueba antes de imputar y escalar. La clasificación estratifica.
  Si se seleccionan modelos mirando repetidamente prueba, se requiere un conjunto final
  adicional o validación anidada. No se afirma desempeño en datos reales.
- Las unidades monetarias del dashboard son ficticias. Los ratios usan sumas, no promedios
  de ratios. Los períodos de los extremos pueden estar incompletos; las metas no son pronósticos.

## Pruebas y calidad

```bash
export UV_CACHE_DIR=/workspace/.cache/uv  # solo necesario en Codex Cloud
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Las pruebas comprueban reproducibilidad, filtros, faltantes, intervalos de Welch contra
una fórmula independiente, agregaciones de DuckDB, métricas, separación entrenamiento/prueba,
estadísticos de imputación y escala aprendidos exclusivamente en entrenamiento, y ejecución
de las páginas mediante `streamlit.testing.v1.AppTest`, incluyendo controles y selecciones vacías.
`.github/workflows/ci.yml` repite estas comprobaciones en cada push y pull request.

### Validación de esta entrega — 8 de octubre de 2026

- Python 3.12.14; instalación con uv y repetición de `uv sync --frozen` correctas.
- **30 pruebas aprobadas**, sin pruebas omitidas; lint y formato Ruff correctos.
- Servidor Streamlit iniciado; `/_stcore/health` devuelve `ok`.
- Chromium ejecutó la navegación de la portada y las cuatro páginas con gráficos
  Plotly, sin excepciones de aplicación ni errores JavaScript durante el recorrido.
  También se verificaron clasificación, correlación Spearman y una descarga real de
  `ventas_filtradas.csv` con los 3.285 registros esperados.
- Una instalación separada con solo `requirements.txt`, sin instalar el paquete local
  ni herramientas de desarrollo, ejecutó la portada y las cuatro demos con AppTest.
- Se guardaron `install_script` y `start_skill` en el borrador de Codex Cloud. Revisa y
  guarda los cambios en la configuración del entorno y publícalo para activarlos.
  Guardar el borrador no equivale a publicar ni a validar una nueva tarea restaurada.
- No se hicieron commits, push, despliegue en Community Cloud ni ejecución remota de
  GitHub Actions. Tampoco se realizó una auditoría completa de accesibilidad o dispositivos móviles.

### Codex Cloud

- Usa el checkout existente `/workspace/datalab`; cada tarea ya está aislada.
- Dependencias y caché se guardan en rutas escribibles; no se modifica el Python del sistema.
- El proceso de Streamlit debe arrancar nuevamente en cada tarea; no sobrevive a snapshots.
- Comprobación del servidor: `curl --fail http://127.0.0.1:8501/_stcore/health` debe devolver `ok`.
  Esto acredita disponibilidad; las pruebas AppTest acreditan los cálculos y los controles.
- La interfaz de onboarding no admite enlaces de vista previa a localhost. Los gráficos
  Plotly se renderizan en un navegador con JavaScript; los tests comprueban su creación,
  pero no sustituyen una inspección visual ni prueban las descargas en un navegador real.
  Usa una instancia desplegada o una vista de navegador que ofrezca la plataforma.
- Los bosques usan 80 árboles y dos hilos; el máximo de observaciones limita el costo de
  la demo. No se necesitan GPU, base de datos persistente, secretos ni dominios de datos externos.
- La red solo se necesita para GitHub y para instalar paquetes públicos desde PyPI.

## Estructura

```text
app.py                  navegación de Streamlit
datalab/data.py          generación de datos sintéticos
datalab/analysis.py      filtros, inferencia, exportación y SQL
datalab/models.py        partición, Pipeline, modelos y evaluación
datalab/ui.py            controles comunes y cachés de presentación
datalab/pages/           portada y cuatro demostraciones
tests/                  pruebas funcionales, estadísticas y de interfaz
```

## Streamlit Community Cloud

1. Integra los archivos en GitHub según las instrucciones siguientes.
2. En [Streamlit Community Cloud](https://share.streamlit.io/), crea una aplicación y
   selecciona el repositorio, la rama publicada y **`app.py`** como archivo principal.
3. En opciones avanzadas elige **Python 3.12**. No configures secretos para estas demos.
4. El repositorio contiene `uv.lock` para instalaciones reproducibles y un
   `requirements.txt` exportado para entornos basados en pip. `.streamlit/config.toml`
   establece el tema y la ejecución sin interfaz gráfica del servidor.
5. Verifica la navegación, filtros, gráficos y descargas tras el despliegue. No se ha
   desplegado automáticamente desde Codex; este paso requiere tu cuenta de Streamlit.

No mantengas manualmente dos listas de dependencias. Después de una actualización deliberada:

```bash
uv lock
uv sync --frozen
uv export --frozen --no-dev --no-emit-project --no-hashes --no-header --output-file requirements.txt
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

`requirements.txt` excluye el paquete local y las herramientas de desarrollo. El paquete
`datalab` está en la raíz, por lo que `app.py` también puede importarlo cuando Community
Cloud instala únicamente los paquetes externos.

## Revisar e integrar en GitHub

El repositorio original no tenía commits ni rama `main`. Los archivos se entregan sin
crear commits ni hacer push. Revisa los archivos nuevos y ejecuta las comprobaciones:

```bash
git status --short
git diff --stat
git diff
git add .
git diff --cached --stat
git diff --cached
git commit -m "feat: construir DataLab con cuatro demos interactivas"
git branch -M main
git push -u origin main
```

Si se han creado commits remotos mientras tanto, primero ejecuta `git fetch origin` y
revisa la historia; no fuerces un push. En un repositorio vacío no existe aún una rama
base para abrir un PR: después del primer commit en `main`, usa ramas de trabajo y PRs
para futuras modificaciones. La publicación requiere permisos de escritura que no se
dan por acreditados por una consulta de lectura a GitHub.
