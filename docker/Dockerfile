# Image: agent_solver
#
# Chromatix and Optiland both require Python >=3.11/3.12 (see
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
# libglu1-mesa and the libx*/libgomp set: an OpenGL/X11 stack that several
# scientific wheels dlopen at import time even when used headlessly. It was
# originally added for gmsh, which CHE-87 removed with JAX-FEM; the packages are
# kept because Optiland's visualization path (vtk 9.6.2) needs the same
# libraries, and removing them to save image layers would trade a measured
# working image for an untested one. Verify against vtk before trimming.

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
