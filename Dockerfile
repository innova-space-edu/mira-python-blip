# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# deps básicos para numpy/pillow/torch + salud del contenedor
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Puerto que expone Flask (lo puedes sobreescribir con PORT en Fly)
ENV PORT=5000

# Cambia EL ARCHIVO.PY por el de cada servicio:
# plot -> plot_service.py
# global -> global_service.py
# ocr -> ocr_service.py
# blip -> blip_service.py
CMD ["python", "blip_service.py"]
