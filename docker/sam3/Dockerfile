ARG CUDA_VER=12.8.0
FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ARG UID
ARG GID
RUN groupadd -g "${GID:?GID build arg required}" developer \
 && useradd -m -u "${UID:?UID build arg required}" -g "${GID}" -s /bin/bash developer

ENV UV_PROJECT_ENVIRONMENT=/opt/venv-sam3 \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    HF_HOME=/opt/hf-cache \
    PYTHONPATH=/workspace/sam3

WORKDIR /build/sam3
# Python pin は親 repo 側 docker/sam3/.python-version で管理
# （sam3 submodule の .gitignore が .python-version を除外しているため fork 汚染を回避）
COPY sam3/pyproject.toml sam3/uv.lock ./
COPY docker/sam3/.python-version ./
# notebooks extra: cv2, matplotlib, decord, einops, pycocotools, scikit-image/learn
# bundled scripts (scripts/qualitative_test.py 等) が cv2/matplotlib を import するため必要
RUN uv sync --locked --no-install-project --extra notebooks \
 && mkdir -p /opt/venv-sam3 /opt/uv-cache /opt/uv-python /opt/hf-cache \
 && chown -R developer:developer /opt/venv-sam3 /opt/uv-cache /opt/uv-python /opt/hf-cache

USER developer
WORKDIR /workspace/sam3
CMD ["bash"]
