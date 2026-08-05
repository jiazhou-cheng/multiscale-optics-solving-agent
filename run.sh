#!/usr/bin/env bash
# Run the agent_solver container, rebuilding only when requested.
#
# `--rebuild` uses Docker's layer cache: it only re-runs `pip install` when
# dependency inputs change. Application source is mounted into the container,
# so ordinary source edits do not require rebuilding the image.
#
# Usage:
#   ./run.sh                 # use existing image + drop into an interactive shell
#   ./run.sh pytest -q       # use existing image + run a command in the container
#   ./run.sh --rebuild ...   # rebuild the image (cached), then run
#   ./run.sh --no-build ...  # explicit alias for the default behavior
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE=agent_solver
DO_BUILD=0

case "${1:-}" in
    --no-build)
        shift
        ;;
    --rebuild)
        DO_BUILD=1
        shift
        ;;
esac

if [[ "$DO_BUILD" -eq 1 ]]; then
    docker build -t "$IMAGE" -f docker/Dockerfile .
fi

# Mount the repo so edits to source/knowledge files are visible immediately
# inside the container (the project is installed with `pip install -e .`).
# Uncomment --gpus all if running on a host with NVIDIA CUDA + nvidia-docker.
docker run --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    "$@"
