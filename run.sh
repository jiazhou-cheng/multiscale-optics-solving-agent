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

# Old container runs created scratch/cache directories as container-owned users.
# Repair known generated paths once; subsequent runs use the invoking host user.
mkdir -p tmp_probes
generated_paths=(tmp_probes)
for path in .pytest_cache .ruff_cache .mypy_cache; do
    if [[ -e "$path" ]]; then
        generated_paths+=("$path")
    fi
done

repair_paths=()
for path in "${generated_paths[@]}"; do
    if [[ ! -w "$path" ]]; then
        repair_paths+=("/workspace/$path")
    fi
done

if [[ "${#repair_paths[@]}" -gt 0 ]]; then
    docker run --rm \
        -v "$(pwd)":/workspace \
        "$IMAGE" \
        chown -R "$(id -u):$(id -g)" "${repair_paths[@]}"
fi

docker_args=(run --rm -i)
if [[ -t 0 ]]; then
    docker_args+=(-t)
fi

# Mount the repo so edits to source/knowledge files are visible immediately
# inside the container (the project is installed with `pip install -e .`).
# Uncomment --gpus all if running on a host with NVIDIA CUDA + nvidia-docker.
docker "${docker_args[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    "$@"
