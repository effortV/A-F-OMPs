ARG PYTHON_BASE_IMAGE=docker.1ms.run/library/python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# Install the stable dependency layer from a minimal valid Hatch package.
# Later edits under app/, ui/ or data/ will no longer reinstall PyTorch.
COPY app/__init__.py ./app/__init__.py
RUN pip install --upgrade pip \
    && pip install --index-url "${PYTORCH_CPU_INDEX_URL}" "torch>=2.2,<3" \
    && pip install .

COPY app ./app
COPY ui ./ui
COPY data ./data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
