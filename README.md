# CanastaCL

Análisis de precios de la **canasta familiar en Chile** a partir de datos de precios al consumidor capturados en supermercados, ferias libres, carnicerías y panaderías de las regiones de Arica y Parinacota, Coquimbo, Valparaíso, Metropolitana, Maule, Ñuble, Biobío, La Araucanía y Los Lagos.

> Proyecto personal de **portafolio en Data Science**. No está asociado a ningún empleador. Su propósito es demostrar competencias en análisis exploratorio, modelado de series temporales y desarrollo de dashboards interactivos.

---

## Inicio del proyecto

Este repositorio marca el inicio del proyecto **CanastaCL**, una iniciativa de portafolio orientada a explorar, modelar y comunicar el comportamiento de los precios de los principales alimentos de la canasta familiar en Chile.

El trabajo se desarrolla de manera iterativa siguiendo un flujo estándar de un proyecto de ciencia de datos:

1. **Ingesta y validación de datos** — carga del dataset crudo, verificación de tipos, codificación y consistencia regional.
2. **Limpieza y preparación** — normalización de unidades, conversión de formatos numéricos al estándar internacional, tratamiento de valores faltantes y atípicos.
3. **Análisis exploratorio (EDA)** — caracterización de precios por región, categoría, tipo de punto de monitoreo y ventana temporal.
4. **Modelado** — construcción de modelos de pronóstico de precios (statsmodels, Prophet) y modelos supervisados (scikit-learn, LightGBM, XGBoost) con interpretación vía SHAP.
5. **Visualización y comunicación** — dashboard interactivo en Streamlit que permita comparar regiones, productos y tendencias.
6. **Reproducibilidad** — entorno aislado con `venv`, dependencias fijadas en `requirements.txt`, control de calidad con `ruff`, `pytest` y `pre-commit`.

El proyecto está estructurado para crecer de forma ordenada: notebooks para exploración, módulos en `src/` para lógica reutilizable, y un dashboard desacoplado en `dashboard/`.

---

## Objetivos

### Objetivos generales
- Entregar un análisis reproducible y bien documentado del comportamiento de los precios de la canasta familiar en Chile.
- Construir una pieza de portafolio que evidencie buenas prácticas de ingeniería y ciencia de datos.

### Objetivos específicos
1. **Caracterizar** la distribución de precios mínimos, máximos y promedios por región, grupo de producto y tipo de punto de monitoreo (supermercado, feria, carnicería, panadería).
2. **Identificar diferencias regionales** significativas en el costo de productos comparables (p. ej. carne bovina, pan, frutas y hortalizas).
3. **Detectar productos volátiles** mediante análisis de dispersión semanal y mensual.
4. **Modelar series de tiempo** de precios promedio para productos representativos y generar pronósticos a 4–12 semanas.
5. **Comparar modelos** estadísticos clásicos (ARIMA/SARIMA) con modelos modernos (Prophet, gradient boosting con features temporales).
6. **Interpretar** las predicciones con SHAP para entender qué variables (región, semana del año, tipo de comercio) impulsan los precios.
7. **Publicar un dashboard** interactivo en Streamlit que permita a un usuario no técnico filtrar por región, categoría y producto, y visualizar tendencias.
8. **Documentar** el proceso de forma clara para que el repositorio pueda ser leído como un caso de estudio.

---

## Dataset

- **Archivo:** `data/raw/precio_consumidor_2026.csv`
- **Volumen:** ~100 mil registros, ~18 MB.
- **Columnas:** `Anio`, `Mes`, `Semana`, `Fecha inicio`, `Fecha termino`, `ID region`, `Region`, `Sector`, `Tipo de punto monitoreo`, `Grupo`, `Producto`, `Unidad`, `Precio minimo`, `Precio maximo`, `Precio promedio`.
- **Cobertura:** 9 regiones de Chile, distintos tipos de comercio, agregación semanal.

---

## Estructura del proyecto

```
CanastaCL/
├── data/
│   ├── raw/              # Datos originales sin modificar
│   └── processed/        # Datos limpios y derivados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_limpieza.ipynb
│   └── 03_modelo.ipynb
├── src/                  # Lógica reutilizable
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   └── model.py
├── dashboard/
│   └── app.py            # Dashboard Streamlit
├── models/               # Modelos entrenados (artefactos)
├── tests/                # Pruebas unitarias
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Stack técnico

| Capa | Herramientas |
|------|--------------|
| Core | pandas, numpy, polars, pyarrow, python-dotenv |
| Ingesta | requests, beautifulsoup4, openpyxl |
| Visualización | matplotlib, seaborn, plotly |
| Geoespacial | geopandas, folium, shapely |
| Series de tiempo | statsmodels, prophet |
| Machine Learning | scikit-learn, lightgbm, xgboost, shap |
| Dashboard | streamlit |
| Calidad | ruff, pytest, pre-commit |

---

## Cómo levantar el proyecto

> Requiere Python 3.11+ en Windows (probado en Windows 11).

```powershell
# 1. Clonar el repositorio
git clone <url-del-repo>
cd CanastaCL

# 2. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) Registrar el venv como kernel de Jupyter
python -m ipykernel install --user --name canastacl --display-name "Python (CanastaCL)"

# 5. Lanzar Jupyter o el dashboard
jupyter lab
# o bien
streamlit run dashboard/app.py
```

---

## Roadmap

- [x] Inicialización del repositorio y estructura base
- [ ] EDA inicial (`notebooks/01_eda.ipynb`)
- [ ] Pipeline de limpieza (`src/data_loader.py`, `notebooks/02_limpieza.ipynb`)
- [ ] Feature engineering (`src/features.py`)
- [ ] Modelo baseline de forecasting (`notebooks/03_modelo.ipynb`)
- [ ] Modelo avanzado + interpretabilidad SHAP
- [ ] Dashboard Streamlit
- [ ] Documentación final y publicación en GitHub

---

## Autor

**Ignacio Tapia** — Proyecto personal de portafolio en Data Science.

## Licencia

Uso educativo y de portafolio. Datos de fuente pública chilena.
