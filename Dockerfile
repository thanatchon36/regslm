# syntax=docker/dockerfile:1.7
# regslm reproduction image (MISSION §0, §6). Canonical path: `make reproduce-docker`.
# Host needs the NVIDIA Container Toolkit (on WSL2: Docker Desktop with the WSL2 GPU backend, or
# nvidia-container-toolkit inside the distro). The image holds code + the three uv envs only;
# model weights (HF cache), data/ and results/ live in volumes (see docker-compose.yml).
#
# Pinned bases (resolved 2026-10-15 with `docker buildx imagetools inspect`):
#   nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04  sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e
#   ghcr.io/astral-sh/uv:0.13.1                 sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336
# torch / vLLM / Unsloth versions are pinned by uv.lock (the three envs resolve separately, D-002).
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential cmake ninja-build git curl ca-certificates \
        make tmux pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app
# Dependency layers first (cache-friendly): lockfile + metadata, then sources.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# Same three envs as `make setup setup-train setup-quant` (the Makefile never syncs one from another).
RUN uv sync --extra serve --extra th --extra pdf \
 && UV_PROJECT_ENVIRONMENT=.venv-train uv sync --extra train --extra th \
 && UV_PROJECT_ENVIRONMENT=.venv-quant uv sync --extra quant --extra th

COPY . .

VOLUME ["/hf"]
ENV PATH="/usr/local/bin:${PATH}"
CMD ["make", "reproduce"]
