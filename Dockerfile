FROM python:3.12-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gnupg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "compute_bfi.py"]
