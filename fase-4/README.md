# Predicción del valor de la vivienda en California – Fase 4 (Monitoreo)

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

## Integrantes

- Joelle Isaac Rios Perez
- Sergio Alejandro Idarraga Agudelo

## Qué se agregó

`monitoreo_estadisticas.py` reutiliza la API de `fase-3` sin modificarla y le agrega dos cosas:

- **Registro.** Escribe una fila por cada llamada a `POST /predict`, incluidas las rechazadas por validación (422). Guarda `fecha_hora` (UTC), `estado_http`, `version` y `prediccion`. En un error, la versión y la predicción quedan vacías. El archivo se abre en modo de agregar, así que el historial se conserva entre ejecuciones.
- **Métricas.** Al ejecutarlo como script, lee el registro y muestra el total de solicitudes y el porcentaje de fallidas (códigos 4xx y 5xx). Con un registro vacío muestra total 0 y «sin solicitudes».

El registro está en `fase-4/data/predicciones_log.csv`. Se puede cambiar con la variable `PREDICTION_LOG_PATH`.

## Ejecución

Desde la **raíz del repositorio**, en Bash:

```bash
docker build -f fase-4/Dockerfile -t modelos-fase4:1.0 .

mkdir -p fase-4/data

docker run -d --name api-fase4 --user "$(id -u):$(id -g)" -p 8000:8000 -e MODEL_VERSION=v1 -v "${PWD}/data/raw:/app/data/raw:ro" -v "${PWD}/fase-2/models:/app/models" -v "${PWD}/fase-4/data:/app/fase-4/data" modelos-fase4:1.0
```

- **Versión:** la API predice con `MODEL_VERSION` (`v1`).
- **Registro:** queda en `fase-4/data/predicciones_log.csv` del equipo anfitrión, montado en `/app/fase-4/data`.

Calcular las métricas con la API corriendo:

```bash
docker exec api-fase4 python monitoreo_estadisticas.py --log data/predicciones_log.csv
```

Sin Docker, desde `fase-4` y con el entorno de `fase-3` activado:

```bash
uvicorn monitoreo_estadisticas:app --port 8000
python monitoreo_estadisticas.py
```

## Reporte

Sobre un registro vacío se enviaron tres solicitudes con `curl` a `http://localhost:8000/predict`:

1. La primera fila de `fase-2/data/ejemplo_prediccion.csv` → 200, 433.644,16 USD.
2. La segunda fila → 200, 452.837,15 USD.
3. La primera fila sin `latitude` → 422, `Field required` en `latitude`.

Registro generado (`fase-4/data/predicciones_log.csv`):

```text
fecha_hora,estado_http,version,prediccion
2026-11-23T03:57:32+00:00,200,v1,433644.16459238174
2026-11-23T03:57:32+00:00,200,v1,452837.1468625378
2026-11-23T03:57:32+00:00,422,,
```

Salida del script:

```text
Total de solicitudes: 3
Porcentaje de solicitudes fallidas: 33.3%
```

**Interpretación.** La API procesó dos sectores con `v1` y rechazó el tercero porque le faltaba `latitude`. La tasa de fallos fue 1/3 (33,3 %). Ese rechazo era el resultado esperado de la prueba de validación, y la falla se corrige en quien envía los datos. Este indicador describe el funcionamiento del servicio, no la calidad de las predicciones, porque no se conoce el valor real de esos sectores.

Al reiniciar el contenedor, el script volvió a mostrar 3 solicitudes: el registro se conserva.

## Versiones

| Versión | Estado | Semilla | MAE | Ubicación |
|---|---|---|---|---|
| `v1` | **Activa** | 42 | 51.424,86 | `fase-2/models/v1/` |
| `v2` | Anterior, disponible | 7 | 34.172,49 | `fase-2/models/v2/` |

Cada carpeta tiene `modelo.joblib`, `metrics.json` y `metadata.json`. Para cambiar de versión, o volver a la anterior, se reinicia el contenedor con otro valor de `MODEL_VERSION` (por ejemplo `-e MODEL_VERSION=v2`). La columna `version` del registro muestra cuál respondió cada solicitud.

Los MAE de `v1` y `v2` no comparan modelos: cada semilla deja distritos distintos en prueba (ver `fase-3/README.md`).

## Política de reentrenamiento

Los datos son del censo de 1990, así que el modelo solo se revisa cuando lleguen sectores nuevos con su valor real (`median_house_value`). Al reunir al menos 500, un integrante del equipo ejecuta `POST /train` para entrenar una candidata con la misma configuración (semilla 42 y partición por zonas). La candidata se activa con `MODEL_VERSION` solo si su MAE, calculado junto con el de la versión activa sobre las mismas zonas reservadas, es menor y además supera al modelo base. Si no mejora, se conserva la versión activa.

## Fuentes

- Datos: [California Housing Prices, Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
- Documentación de [FastAPI](https://fastapi.tiangolo.com/).
- La API y el modelo son los de `fase-3` y `fase-2` de este repositorio.
