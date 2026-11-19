# CPU-only API image. No PyTorch: inference uses the exported ONNX model.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TINY_VOCA_LANGUAGE=ar \
    TINY_VOCA_FRONTEND=rules

WORKDIR /app

# No apt step: onnxruntime, soundfile and numpy wheels bundle their native
# libraries. This also keeps the build independent of Debian mirror access.
# Optional voice-service extra only: the core library keeps torch out of this image.
RUN pip install --no-cache-dir \
      "fastapi>=0.110,<1" \
      "uvicorn[standard]>=0.27,<1" \
      "onnxruntime>=1.17,<2" \
      "httpx>=0.27,<1" \
      "soundfile>=0.12,<1" \
      "numpy>=1.24,<3" \
      "camel-tools>=1.6,<2"

# The ONNX model, its config, the G2P file and the chat page are shipped in server/.
COPY server/ ./server/

RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# CALIMA MSA database, downloaded once as appuser so it lives in that user's
# data directory and is readable at runtime. Never at every container start.
RUN camel_data -i defaults

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "server.tts_service:app", "--host", "0.0.0.0", "--port", "8000"]
