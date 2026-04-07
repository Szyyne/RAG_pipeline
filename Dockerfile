# RAG Pipeline Doctor — Dockerfile
# Builds a containerized FastAPI server exposing the OpenEnv RL environment.
# HuggingFace Spaces requires port 7860.

FROM python:3.11-slim

# Non-root user for HF Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY models.py tasks.py client.py inference.py openenv.yaml ./
COPY server/ ./server/

# Give appuser ownership
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true

# Health check — pings /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
