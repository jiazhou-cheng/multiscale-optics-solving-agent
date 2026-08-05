# Image: agent_solver
#
# Chromatix, SAX, and Optiland all require Python >=3.11/3.12 (see
# docker/requirements.txt header for per-package install hazards). This
# image exists because the host environment only had Python 3.10/3.8
# available. python:3.12-slim satisfies every pinned solver's
# requires-python floor simultaneously.
#
# Layering is deliberate for build-cache reuse: the dependency install (slow,
# network-heavy) is its own layer keyed only on requirements.txt. Editing
# application source under src/ never invalidates that layer, so
# `docker build` after a source-only change reinstalls nothing.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libglu1-mesa \
        libgl1 \
        libxrender1 \
        libxcursor1 \
        libxinerama1 \
        libxft2 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# libglu1-mesa etc.: jax-fem's `generate_mesh` module imports `gmsh` even for
# headless meshing, and gmsh's Python bindings dlopen a real OpenGL/X11
# stack at import time (`OSError: libGLU.so.1: cannot open shared object
# file`) -- see knowledge/solvers/jax_fem/failure_guide.md.

WORKDIR /workspace

# --- dependency layer (cache key: requirements.txt only) ---
# torch is pinned and installed from the CPU-only wheel index first: the
# default PyPI index resolves plain `torch` to a CUDA build that pulls in
# several GB of nvidia-* packages even though this image has no GPU.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# --- source layer (cache key: everything else) ---
COPY . /workspace
RUN pip install --no-cache-dir --no-deps -e .

CMD ["bash"]
