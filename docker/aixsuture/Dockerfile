ARG CUDA_VER=11.8.0
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ARG UID=1002
ARG GID=1002
RUN groupadd -g ${GID} developer \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash developer

ENV UV_PROJECT_ENVIRONMENT=/opt/venv-aixsuture \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1

WORKDIR /build/aixsuture
COPY aixsuture/pyproject.toml aixsuture/uv.lock aixsuture/.python-version ./
RUN uv sync --locked --no-install-project \
 && mkdir -p /opt/venv-aixsuture /opt/uv-cache /opt/uv-python \
 && chown -R developer:developer /opt/venv-aixsuture /opt/uv-cache /opt/uv-python

USER developer
WORKDIR /workspace/aixsuture
CMD ["bash"]
