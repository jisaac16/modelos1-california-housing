FROM python:3.14-slim

WORKDIR /app

# LightGBM necesita libgomp1 para ejecutarse.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
