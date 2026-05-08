FROM python:3.11-slim

ARG DISABLE_GPU=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    DISABLE_GPU=${DISABLE_GPU} \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-linux-cpu.txt ./
RUN python -m pip install --upgrade pip \
    && if [ "$DISABLE_GPU" = "true" ]; then \
        python -m pip install --no-cache-dir -r requirements-linux-cpu.txt; \
    else \
        python -m pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app ./app
COPY main.py .

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
