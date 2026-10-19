# ---- build stage: install dependencies into a virtualenv ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---- runtime stage ----
FROM python:3.11-slim
LABEL org.opencontainers.image.title="heart-disease-risk-api" \
      org.opencontainers.image.description="FastAPI service for the UCI heart disease classifier"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/heart_model.joblib \
    METADATA_PATH=/app/models/model_metadata.json
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY api/ ./api/
COPY models/heart_model.joblib models/model_metadata.json ./models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
