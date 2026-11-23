# Se construye desde la raíz del repositorio:
#   docker build -f fase-4/Dockerfile -t modelos-fase4:1.0 .
FROM python:3.14-slim

WORKDIR /app

# LightGBM necesita libgomp1 para ejecutarse.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

# Código de fase-2 y fase-3 que reutiliza el monitoreo. Datos, modelos y registro llegan por montajes.
COPY fase-2/src fase-2/src
COPY fase-2/train.py fase-2/predict.py fase-2/
COPY fase-3/api.py fase-3/
COPY fase-4/monitoreo_estadisticas.py fase-4/

ENV MODELS_DIR=/app/models \
    DATA_PATH=/app/data/raw/housing.csv \
    MODEL_VERSION=v1

WORKDIR /app/fase-4
EXPOSE 8000
CMD ["uvicorn", "monitoreo_estadisticas:app", "--host", "0.0.0.0", "--port", "8000"]
