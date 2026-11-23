# Predicción del valor de la vivienda en California – Fase 3 (API REST)

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

## Integrantes

- Joelle Isaac Rios Perez
- Sergio Alejandro Idarraga Agudelo

## Relación con la Fase 2

La API usa el mismo modelo (LightGBM Reducido), la misma partición por zonas y las mismas versiones guardadas. No copia código: importa de `fase-2/` la carga del modelo (`predict.cargar_modelo`), la validación de tipos (`src/procesamiento.py`) y el entrenamiento (`train.entrenar`). La preparación de los datos viene dentro de cada `modelo.joblib`.

## Rutas

| Ruta | Qué hace | Respuesta |
|---|---|---|
| `GET /health` | Indica que la API está en ejecución | 200 `{"estado": "en ejecución"}` |
| `POST /predict` | Predice el valor de un sector con la versión configurada | 200 con `prediccion`, `version` y `aviso` |
| `POST /train` | Entrena una versión nueva con los datos y la configuración de la fase 2 | 201 con `version`, `ruta` y `metricas` |

Errores de `/predict`:

| Caso | Código |
|---|---|
| Falta un campo o un número llega como texto | 422, el mensaje indica el campo |
| `households = 0` | 422 |
| La versión configurada no existe | 503, con la carpeta donde buscar versiones |

## Campos de entrada

Todos son obligatorios. Un dato faltante se envía como `null` y el modelo lo trata como faltante, igual que en la fase 2. La variable objetivo no se envía.

| Campo | Tipo | Unidad |
|---|---|---|
| `longitude`, `latitude` | número | grados decimales |
| `housing_median_age` | número | años (mediana del sector) |
| `total_rooms` | número | cuartos del sector |
| `total_bedrooms` | número | habitaciones del sector (el modelo no la usa) |
| `population` | número | personas |
| `households` | número, distinto de 0 | hogares |
| `median_income` | número | decenas de miles de USD (8.3 = 83.000 USD) |
| `ocean_proximity` | texto | `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY` o `NEAR OCEAN` |

Una categoría distinta de `ocean_proximity` se trata como faltante. Los campos adicionales se ignoran. La `prediccion` está en dólares de 1990 y solo es válida para sectores por debajo del tope censal de 500.001 USD.

## Configuración y selección de versión

| Variable | Por defecto (local) | Uso |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión que usa `/predict` |
| `MODELS_DIR` | `fase-2/models` | Carpeta de versiones; `/train` guarda allí |
| `DATA_PATH` | `data/raw/housing.csv` | Datos para `/train` |
| `SEED` | `42` | Semilla de `/train` |

Para usar otra versión, se reinicia la API con `MODEL_VERSION` igual al nombre de una carpeta de `MODELS_DIR` (por ejemplo `v2`). `/train` acepta un cuerpo vacío y crea el siguiente identificador libre (`v3`, `v4`...). También acepta `{"version": "...", "seed": ...}`. Nunca sobrescribe una versión.

## Ejecución local

Comandos para **Bash**, desde la carpeta `fase-3`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --port 8000
```

En otra terminal, también desde `fase-3`:

```bash
python cliente.py
curl -X POST localhost:8000/train
```

## Docker

Desde la **raíz del repositorio**, en Bash. La imagen se construye desde la raíz porque incluye el código de `fase-2`:

```bash
docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 .

docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 -e MODEL_VERSION=v1 -e COMMIT=$(git rev-parse HEAD) -v "${PWD}/data/raw:/app/data/raw:ro" -v "${PWD}/fase-2/models:/app/models" modelos-fase3:1.0
```

- **Puerto:** 8000.
- **Datos:** `data/raw` se monta en `/app/data/raw`, solo lectura.
- **Modelos:** `fase-2/models` se monta en `/app/models`. Las versiones que crea `/train` quedan allí, en el equipo anfitrión.
- **Commit:** `COMMIT` se registra en el `metadata.json` de las versiones nuevas, porque la imagen no trae Git.
- **Linux:** `--user` deja los archivos a nombre del usuario. En PowerShell se omite esa opción.

Con el contenedor corriendo, desde la raíz: `python fase-3/cliente.py`.

## Pruebas

Desde `fase-3`, con el entorno activado:

```bash
python -m pytest
```

Son 7 pruebas que usan la versión `v1` guardada:

- `/health` responde 200.
- Una entrada válida devuelve la predicción y la versión, con el mismo valor que daba `predict.py` en la fase 2: 433.644,16 USD.
- Una entrada sin `latitude` se rechaza con 422.
- Un texto en un número y `households = 0` se rechazan.
- Un `null` permitido predice.
- Una versión inexistente da 503.

`/train` se comprobó a mano (ver evidencia).

## Versiones

| Versión | Semilla | MAE | R² | Commit | Python |
|---|---|---|---|---|---|
| `v1` | 42 | 51.424,86 | 0,5752 | `f873e21` | 3.14.2 |
| `v2` | 7 | 34.172,49 | 0,7421 | `f873e21` | 3.14.2 |

El modelo, las métricas, el entorno y el commit de cada versión están en `fase-2/models/<versión>/` (`modelo.joblib`, `metrics.json` y `metadata.json`).

**Las métricas de `v1` y `v2` no comparan modelos.** Cada semilla deja en prueba distritos distintos, así que el MAE menor de `v2` se debe a la muestra y no a un mejor modelo. Las versiones que crea `/train` siguen la misma regla: solo se comparan si usan la misma semilla y los mismos datos.

## Evidencia de ejecución

Docker en Bash (Artix Linux), con `MODEL_VERSION=v1`.

Cliente (`python fase-3/cliente.py`):

```text
GET /health -> 200 {'estado': 'en ejecución'}
Solicitud: {"longitude": -122.23, "latitude": 37.85, "housing_median_age": 41.0, "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0, "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}
POST /predict -> 200 {"prediccion": 433644.16459238174, "version": "v1", "aviso": "Válido para sectores censales con valor menor a 500001 USD (censo de 1990). No estima viviendas individuales ni el mercado actual."}
```

Solicitud sin `latitude`:

```text
422 {"detail":[{"type":"missing","loc":["body","latitude"],"msg":"Field required", ...}]}
```

Versión inexistente (`MODEL_VERSION=v99`):

```text
503 {"detail":"Modelo 'v99' no disponible: El archivo de modelo /app/models/v99/modelo.joblib no existe. Configure MODEL_VERSION con una versión de /app/models."}
```

`curl -X POST localhost:8000/train`:

```text
201 {"version":"v3","ruta":"/app/models/v3","metricas":{"mae_prueba":51424.86, "r2_prueba":0.5752, "baseline_mae":86373.73, ...}}
```

`v3` quedó en `fase-2/models/v3` del equipo anfitrión. Con la semilla 42 da las mismas métricas de `v1`. Git no la sube, porque solo se versionan `v1` y `v2`.

Pruebas dentro del contenedor: `7 passed`.

## Fuentes

- Datos: [California Housing Prices, Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
- Documentación de [FastAPI](https://fastapi.tiangolo.com/), [scikit-learn](https://scikit-learn.org/stable/) y [LightGBM](https://lightgbm.readthedocs.io/).
- El código de entrenamiento, predicción y preparación es el de `fase-2` de este repositorio.
