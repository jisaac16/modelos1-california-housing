# Se construye desde la raíz del repositorio para incluir el código de fase-2:
#   docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 .
FROM python:3.14-slim

WORKDIR /app

# LightGBM necesita libgomp1 para ejecutarse.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

# Código reutilizado de fase-2 y la API. Datos y modelos llegan por montajes.
COPY fase-2/src fase-2/src
COPY fase-2/train.py fase-2/predict.py fase-2/
COPY fase-2/data fase-2/data
COPY fase-3 fase-3

ENV MODELS_DIR=/app/models \
    DATA_PATH=/app/data/raw/housing.csv \
    MODEL_VERSION=v1

WORKDIR /app/fase-3
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
