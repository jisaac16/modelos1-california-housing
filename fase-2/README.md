# Predicción del valor de la vivienda en California – Fase 2

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

**Versión entregada del código:** etiqueta de Git `entrega-2`.

## Integrantes

- Joelle Isaac Rios Perez
- Sergio Alejandro Idarraga Agudelo

## Problema y relación con la Fase 1

Se estima el valor mediano de la vivienda (`median_house_value`, en dólares de 1990) de un sector censal de California. Es un problema de regresión. Se conservan los datos y el modelo de la Fase 1 (LightGBM Reducido), y el notebook se convierte en una aplicación que entrena y predice por comandos.

## Correcciones respecto a la Fase 1

| Retroalimentación | Cambio realizado |
|---|---|
| Cargar solo `modelo.joblib` no bastaba: la preparación dependía de variables del notebook | La preparación está en `src/procesamiento.py` y se guarda con el modelo en un `Pipeline` |
| El conjunto de prueba se usó para escoger el modelo | `comparar_modelos.py` compara las variantes solo dentro del entrenamiento. La prueba se reserva para la evaluación final |
| La división aleatoria no separaba zonas geográficas | Se separan zonas completas de 0,5 grados con `GroupShuffleSplit` |
| Faltaba definir la respuesta si `households` es cero | `predict.py` termina con un error claro e indica las filas |
| El tope de 500.001 USD debía ser explícito en la aplicación | `predict.py` lo avisa en cada ejecución y queda en `metadata.json` |
| Un Wilcoxon no significativo no demuestra equivalencia | La versión reducida se justifica por razones prácticas: error casi igual con menos variables. Los contrastes por pares de la Fase 1 se consideran exploratorios |
| El notebook anuncia variantes OOB y selección paso a paso que el código no hace | Aclaración: solo se compararon dos listas de variables prefijadas. La carpeta `fase-1` se conserva sin cambios |

Estos cambios afectan la partición y las métricas: no son comparables directamente con las de la Fase 1 (ver [Resultados](#resultados)).

## Instalación

Versión de Python: **3.14**. Comandos para **PowerShell**, desde la carpeta `fase-2`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Bash, el entorno se activa con `source .venv/bin/activate`.

## Datos

- **Fuente:** [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices). No requiere credenciales porque el archivo está en el repositorio.
- **Entrenamiento:** `../data/raw/housing.csv`, en la raíz del repositorio. Es el mismo archivo de la Fase 1.
- **Ejemplo para predecir:** `data/ejemplo_prediccion.csv`.

Esquema del CSV de predicción (sin la variable objetivo):

| Columna | Tipo | ¿Puede venir vacía? | ¿La usa el modelo? |
|---|---|---|---|
| `longitude`, `latitude` | número | sí | sí |
| `housing_median_age` | número | sí | sí |
| `total_rooms` | número | sí | sí |
| `total_bedrooms` | número | sí | no |
| `population` | número | sí | solo en `population_per_household` |
| `households` | número, distinto de 0 | sí | solo en las razones por hogar |
| `median_income` | número | sí | sí |
| `ocean_proximity` | texto | sí | sí |

Comportamiento de `predict.py`:

- **Columna ausente, texto en una columna numérica o `households = 0`:** error claro, código de salida 1 y sin archivo de salida. Las filas del mensaje se cuentan desde 0, sin el encabezado.
- **Celdas vacías:** LightGBM las trata como faltantes y predice. No se eliminan filas.
- **Columnas adicionales:** se ignoran con un aviso y se conservan en la salida.
- **Categorías nuevas de `ocean_proximity`:** se tratan como faltantes, con un aviso.
- **Salida:** las columnas de entrada, en el mismo orden de filas, más la columna `prediccion` en dólares.

## Uso

```powershell
python train.py --help
python predict.py --help
```

Entrenar una versión:

```powershell
python train.py --data ../data/raw/housing.csv --target median_house_value --output-dir models --version v3 --seed 42
```

Si la versión ya existe, el script termina con error y no cambia nada. También acepta `--test-size` (por defecto `0.20`).

Predecir con una versión guardada, sin reentrenar:

```powershell
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

Repetir la comparación de modelos (tarda varios minutos):

```powershell
python comparar_modelos.py --data ../data/raw/housing.csv
```

## Ubicación de los archivos

```
fase-2/
├── train.py, predict.py     # Puntos de entrada
├── comparar_modelos.py      # Elección del modelo sin usar la prueba
├── comparacion_modelos.csv  # Resultado de la comparación
├── src/procesamiento.py     # Validación y preparación compartidas
├── models/v1, models/v2     # modelo.joblib, metrics.json, metadata.json
├── data/                    # CSV de ejemplo
├── outputs/                 # Predicciones generadas (no se suben a Git)
├── tests/                   # Pruebas
└── requirements.txt, Dockerfile, .dockerignore
```

Las versiones usan el mismo algoritmo con distinta semilla, y con ella cambian las zonas de prueba. `v2` demuestra el versionamiento; no se escogió por su métrica.

| Versión | Semilla | Entrenamiento / prueba | MAE | RMSE | R² | MAE base |
|---|---|---|---|---|---|---|
| `v1` | 42 | 15.718 / 3.957 | 51.424,86 | 67.281,28 | 0,5752 | 86.373,73 |
| `v2` | 7 | 15.781 / 3.894 | 34.172,49 | 45.468,46 | 0,7421 | 70.316,97 |

## Docker

Comandos para **Bash** (Linux), desde `fase-2`. Son los que se ejecutaron para la evidencia, en Artix Linux con Docker 29.9:

```bash
docker build -t modelos-fase2:1.0 .

mkdir -p models outputs

docker run --rm --user "$(id -u):$(id -g)" -e COMMIT=$(git rev-parse HEAD) -v "${PWD}/../data/raw:/app/data/raw:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data data/raw/housing.csv --target median_house_value --output-dir models --version v-docker --seed 42

docker run --rm --user "$(id -u):$(id -g)" -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

En Linux, `--user` deja los archivos generados a nombre del usuario; sin esa opción quedan a nombre de `root`.

Los mismos comandos para **PowerShell**, desde `fase-2`. Esta variante no se ejecutó, porque el equipo Windows del proyecto no tiene WSL y Docker Desktop no arranca en él:

```powershell
docker build -t modelos-fase2:1.0 .

New-Item -ItemType Directory -Force models, outputs

docker run --rm -e COMMIT=$(git rev-parse HEAD) -v "${PWD}/../data/raw:/app/data/raw:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data data/raw/housing.csv --target median_house_value --output-dir models --version v-docker --seed 42

docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

Los modelos y las predicciones quedan en `models/` y `outputs/` del equipo anfitrión, aunque el contenedor se elimine. La imagen no trae Git, así que el hash del commit se pasa con `COMMIT`; sin esa variable, `metadata.json` indica `"no disponible"`.

## Pruebas

Con el entorno activado, desde `fase-2`:

```powershell
python -m pytest
```

Hay 14 pruebas que usan datos sintéticos y carpetas temporales. Cubren el entrenamiento y sus artefactos, la partición por zonas, la carga y la predicción, los valores faltantes, las entradas inválidas y el versionamiento.

## Resultados

**Elección del modelo.** MAE con validación cruzada de 5 particiones por zonas, usando solo las 15.718 filas de entrenamiento:

| Variante | MAE promedio | Desviación |
|---|---|---|
| **LightGBM (Reducido)** | **36.853,22** | 4.287,63 |
| LightGBM (Completo) | 36.908,71 | 4.418,33 |
| Random Forest (Completo) | 37.893,93 | 4.478,49 |
| Random Forest (Reducido) | 37.901,69 | 4.571,12 |
| SVM (Completo) | 45.910,47 | 6.681,56 |
| SVM (Reducido) | 46.788,75 | 7.652,85 |
| Dummy (base) | 76.797,92 | 3.341,75 |

La versión reducida de LightGBM se mantiene: su diferencia con la completa (unos 55 USD) es muy pequeña frente a la variación entre particiones, y usa menos variables. Esto no demuestra que sean equivalentes. Tampoco la ventaja sobre Random Forest, de cerca de 1.000 USD, supera esa variación.

**Evaluación final (`v1`).** Prueba en zonas no vistas: MAE de 51.424,86 USD frente a 86.373,73 USD del modelo base, una reducción del 40,5 %.

**El resultado depende mucho de las zonas que caen en prueba.** En una comprobación aparte con 8 semillas:

| Partición | MAE promedio | Mínimo | Máximo |
|---|---|---|---|
| Aleatoria (la de la Fase 1) | 28.378 | 27.767 | 29.229 |
| Por zonas | 41.624 | 33.248 | 54.980 |

La semilla 42 quedó en la parte alta del rango. Se mantuvo porque cambiarla por una con mejor resultado sesgaría la evaluación. La partición aleatoria da un error menor porque muchos sectores de prueba tienen vecinos en entrenamiento; la partición por zonas representa mejor una zona nueva.

## Limitaciones conocidas

- Los 965 sectores en el tope de 500.001 USD se excluyen del entrenamiento, y su valor real se desconoce. El modelo no aplica a sectores por encima del tope, y no se puede saber de antemano si un sector nuevo lo superaría.
- La unidad de predicción es el sector censal de 1990: no estima viviendas individuales ni el mercado actual.
- `modelo.joblib` necesita la carpeta `src/` y versiones compatibles de las librerías para cargarse. Solo deben cargarse archivos de confianza.
- Con los mismos datos y semilla, `v1` (Windows, Python 3.14.2) y su repetición en Docker (Linux, Python 3.14.10) dieron las mismas métricas y predicciones. Podrían variar con otras versiones de las librerías.
- La imagen `python:3.14-slim` fija la versión 3.14, pero no el parche de Python: puede cambiar al reconstruirla. La versión usada queda en `metadata.json`.

## Evidencia de ejecución

Entrenamiento de `v1` (en PowerShell):

```text
Observaciones totales: 20640. Sin valores censurados: 19675.
Entrenando modelo LightGBM Reducido...
Entrenamiento: 15718 filas. Prueba: 3957 filas.
Modelo -> MAE: 51,424.86, RMSE: 67,281.28, R2: 0.5752
Base   -> MAE: 86,373.73, RMSE: 106,756.40, R2: -0.0695
Archivos guardados en models\v1: modelo.joblib, metrics.json, metadata.json
```

Predicción con `v1` sobre el archivo de ejemplo:

```text
2 predicciones guardadas en outputs/predicciones_v1.csv
prediccion: 433644.16, 452837.15
```

Entrada inválida (sin `latitude`), con código de salida 1 y sin archivo de salida:

```text
Error: Faltan columnas requeridas: latitude
```

Pruebas:

```text
14 passed
```

Docker (en Bash), con la imagen `modelos-fase2:1.0`. Entrenamiento con la semilla de `v1`, que da las mismas métricas:

```text
Entrenamiento: 15718 filas. Prueba: 3957 filas.
Modelo -> MAE: 51,424.86, RMSE: 67,281.28, R2: 0.5752
Base   -> MAE: 86,373.73, RMSE: 106,756.40, R2: -0.0695
Archivos guardados en models/v-docker: modelo.joblib, metrics.json, metadata.json
```

Predicción, con los contenedores ya eliminados y los archivos en `models/` y `outputs/` del equipo anfitrión:

```text
2 predicciones guardadas en outputs/predicciones.csv
prediccion: 433644.16, 452837.15
```

Errores dentro del contenedor, ambos con código de salida 1 (y sin archivo de salida en el segundo):

```text
Error: La versión 'v-docker' ya existe en models/v-docker. No se sobrescribirá.
Error: Faltan columnas requeridas: latitude
```

Pruebas dentro del contenedor (`python -m pytest`):

```text
14 passed
```

## Fuentes

- Datos: [California Housing Prices, Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
- Documentación de [scikit-learn](https://scikit-learn.org/stable/) y [LightGBM](https://lightgbm.readthedocs.io/).
- Las reglas de preparación se adaptaron del notebook de la Fase 1 de este repositorio.
