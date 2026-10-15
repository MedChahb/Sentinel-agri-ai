# Build from the repository root: docker build -f api/Dockerfile -t sentinel-agri-api .
FROM python:3.11-slim

WORKDIR /app

# rasterio's bundled GDAL needs libexpat, and xgboost needs libgomp (OpenMP); the slim image has neither.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# The gap-filling code of part 2 (pure numpy / pandas), used by /gapfill.
COPY src/sentinel_agri/__init__.py src/sentinel_agri/config.py src/sentinel_agri/gapfill.py src/sentinel_agri/
ENV PYTHONPATH=/app/src

COPY api/ api/
# Exported models: models/eurosat_resnet18.onnx (+ .json) and models/gapfill/. Without them, the
# matching endpoint answers 503 and the rest of the API still works.
COPY models/ models/

ENV MODEL_PATH=/app/models/eurosat_resnet18.onnx \
    GAPFILL_DIR=/app/models/gapfill
EXPOSE 8000

# AWS Lambda Web Adapter: on Lambda, it turns each request into an HTTP call to uvicorn, so the same
# image runs on Lambda, ECS or a laptop. Outside Lambda, it is never started.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
