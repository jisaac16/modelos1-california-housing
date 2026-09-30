# Predicción del valor de la vivienda en California – Fase 1

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

## Integrantes

- Joelle Isaac Rios Perez
- Sergio Alejandro Idarraga Agudelo

## Descripción del problema

A partir de datos del censo de California de 1990 se busca estimar el valor de la vivienda en cada sector censal (grupo de bloques), usando información demográfica (población, hogares, ingreso), estructural (habitaciones, dormitorios, antigüedad) y geográfica (coordenadas y cercanía al océano).

Es un problema de **aprendizaje supervisado de regresión**: la variable objetivo es continua (dólares).

## Fuente del conjunto de datos

*California Housing Prices* (Kaggle): https://www.kaggle.com/datasets/camnugent/california-housing-prices

- 20,640 sectores censales.
- 9 variables predictoras (8 numéricas y 1 categórica, `ocean_proximity`) y 1 variable objetivo.
- `total_bedrooms` tiene 207 valores faltantes (1 %).
- Archivo en el repositorio: `data/raw/housing.csv`.

## Objetivo del modelo

Predecir el valor mediano de las viviendas de un sector (`median_house_value`).

## Algoritmo utilizado

Se compararon un modelo base (`DummyRegressor`, que predice siempre la media) y tres algoritmos, cada uno en una variante completa y una reducida (sin variables redundantes): **LightGBM**, **Random Forest** y **SVM (SVR)**.

El modelo final es **LightGBM (Reducido)**: el de mejor desempeño en prueba, con sobreajuste moderado y tiempos de entrenamiento y predicción bajos.

## Métricas empleadas

- **MAE**: error promedio en dólares, fácil de interpretar.
- **RMSE**: penaliza más los errores grandes; comparado entre entrenamiento y prueba permite medir el sobreajuste.
- **R²**: proporción de la variabilidad del valor explicada por el modelo.

Además, se aplicaron pruebas de Wilcoxon sobre los errores absolutos para verificar si las diferencias entre modelos son estadísticamente significativas.

## Principales resultados

Resultados en el conjunto de prueba (20 %):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (baseline) | 80,134.51 | 99,865.12 | -0.0002 |
| **LightGBM (Reducido)** | **29,228.51** | **42,860.75** | **0.8158** |
| Random Forest (Reducido) | 30,465.51 | 46,004.68 | 0.7877 |
| SVM (Reducido) | 45,238.60 | 64,734.64 | 0.5797 |

- LightGBM (Reducido) reduce el MAE del modelo base en un 63.5 % y explica el 81.6 % de la variabilidad.
- Eliminar las variables redundantes no afectó significativamente el desempeño de LightGBM (Wilcoxon, p = 0.23).
- Random Forest presenta un sobreajuste marcado (R² de 0.97 en entrenamiento frente a 0.79 en prueba). Su diferencia de error con LightGBM (Reducido) no es estadísticamente significativa (Wilcoxon, p = 0.11), así que la elección de LightGBM se justifica sobre todo por su menor sobreajuste y sus tiempos de entrenamiento y predicción mucho más bajos.

**Limitación:** los sectores con valor en el tope del censo ($500,001) se excluyeron antes de dividir los datos, así que estos resultados aplican a sectores por debajo de ese valor.

El modelo final se guarda en `modelo.joblib`.

## Estructura

```
fase-1/
├── notebook.ipynb   # Análisis, preparación, modelos, evaluación y guardado
├── modelo.joblib    # Modelo final entrenado (LightGBM Reducido)
└── README.md
data/raw/housing.csv # Conjunto de datos
requirements.txt     # Dependencias con versiones fijas
```

## Instrucciones para ejecutar el notebook

Requiere Python 3.14.

1. Clonar el repositorio y entrar a la carpeta:
   ```bash
   git clone git@github.com:jisaac16/modelos1-california-housing-work.git
   cd modelos1-california-housing-work
   ```
2. Crear un entorno virtual e instalar las dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Abrir el notebook desde la carpeta `fase-1` (usa rutas relativas al dataset):
   ```bash
   cd fase-1
   jupyter notebook notebook.ipynb
   ```
4. Ejecutar todas las celdas en orden (*Run All*). El notebook se ejecuta de principio a fin sin intervención manual, usa la semilla `42` y genera `modelo.joblib`.

Para usar el modelo guardado sin reentrenar:

```python
import joblib
modelo = joblib.load("modelo.joblib")
```

Los datos nuevos deben prepararse igual que en el entrenamiento; ver la función `preparar_para_prediccion` en la sección *Almacenamiento del modelo* del notebook.
