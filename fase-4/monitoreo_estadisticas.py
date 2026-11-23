#!/usr/bin/env python3
# Monitoreo y métricas de /predict para Fase 4.
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

# Agrega fase-3 al path invocar la API.
ruta_fase3 = Path(__file__).resolve().parents[1] / "fase-3"
if str(ruta_fase3) not in sys.path:
    sys.path.insert(0, str(ruta_fase3))

import api as fase3_api  # noqa: E402

app = fase3_api.app

# Columnas y ruta por defecto del archivo CSV de monitoreo.
encabezados_log = ["fecha_hora", "estado_http", "version", "prediccion"]
ruta_log_predicciones = Path(__file__).resolve().parent / "data" / "predicciones_log.csv"


# Obtiene la ruta del log desde variable de entorno o usa la ruta por defecto.
def get_log_path() -> Path:
    custom = os.environ.get("PREDICTION_LOG_PATH", "").strip()
    return Path(custom).expanduser() if custom else ruta_log_predicciones


# Agrega una fila al CSV conservando el historial entre ejecuciones.
def registrar_evento(
    estado_http: int,
    version: str | None = None,
    prediccion: float | int | str | None = None,
) -> None:
    log_path = get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    escribir_cabecera = (not log_path.exists()) or (log_path.stat().st_size == 0)

    with log_path.open("a", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        if escribir_cabecera:
            writer.writerow(encabezados_log)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            int(estado_http),
            "" if version is None else str(version),
            "" if prediccion is None else str(prediccion),
        ])


# Registra rechazos 422 de validación en /predict antes de ejecutar predicción.
@app.exception_handler(RequestValidationError)
async def manejador_validacion_fase4(request: Request, exc: RequestValidationError):
    if request.url.path.rstrip("/") == "/predict":
        registrar_evento(estado_http=422)
        request.state.monitoreo_registrado = True
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Intercepta llamadas a POST /predict y registra el resultado final de la solicitud.
@app.middleware("http")
async def middleware_monitoreo_predict(request: Request, call_next):
    if request.url.path.rstrip("/") != "/predict" or request.method.upper() != "POST":
        return await call_next(request)

    response = await call_next(request)
    if getattr(request.state, "monitoreo_registrado", False):
        return response

    # Lee el cuerpo para extraer versión y predicción cuando la respuesta es 200.
    body_bytes = b"".join([chunk async for chunk in response.body_iterator])
    version = ""
    prediccion = ""
    if response.status_code == 200:
        try:
            data = json.loads(body_bytes.decode("utf-8"))
            version = data.get("version", "")
            prediccion = data.get("prediccion", "")
        except Exception:
            pass

    registrar_evento(estado_http=response.status_code, version=version, prediccion=prediccion)
    return Response(
        content=body_bytes,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


# Calcula el total de solicitudes y el porcentaje de fallos HTTP 4xx/5xx.
def calcular_metricas(log_path: str | Path = ruta_log_predicciones) -> dict[str, object]:
    path = Path(log_path).expanduser()
    total = 0
    fallidas = 0

    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                estado_raw = (fila.get("estado_http") or "").strip()
                if not estado_raw:
                    continue
                total += 1
                try:
                    codigo = int(estado_raw)
                except ValueError:
                    fallidas += 1
                    continue
                if 400 <= codigo <= 599:
                    fallidas += 1

    if total == 0:
        return {"total_solicitudes": 0, "fallidas": 0, "porcentaje_fallos": "sin solicitudes"}

    porcentaje = (fallidas / total) * 100.0
    return {
        "total_solicitudes": total,
        "fallidas": fallidas,
        "porcentaje_fallos": f"{porcentaje:.1f}%",
    }


# Ejecuta el cálculo de métricas por consola e imprime los resultados.
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calcula total de solicitudes y % de fallos 4xx/5xx.")
    parser.add_argument("--log", default=str(ruta_log_predicciones), help="Ruta al CSV de registro.")
    args = parser.parse_args(argv)

    res = calcular_metricas(args.log)
    print(f"Total de solicitudes: {res['total_solicitudes']}")
    print(f"Porcentaje de solicitudes fallidas: {res['porcentaje_fallos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())