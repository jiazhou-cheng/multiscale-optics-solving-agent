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
#   ./run.sh --gpu ...       # use the CUDA image (agent_solver_gpu) with GPUs attached
#
# GPU mode (CHE-60 / PB4a)
# ------------------------
# `--gpu` switches to the separately-built `agent_solver_gpu` image (see
# docker/Dockerfile.gpu) and attaches devices via docker's `--gpus`. It is
# opt-in: without the flag nothing about the CPU path changes, and no GPU is
# visible inside the container even on a GPU host.
#
# Device selection honors the "GPU server resource policy" in AGENTS.md: this is
# a shared host, so at most 2 GPUs may be used, and visibility is configured
# through the container rather than host state. Select devices with MOA_GPUS,
# which takes docker `--gpus` device syntax and defaults to a single GPU:
#
#   ./run.sh --gpu pytest -q -m gpu              # device=0 (default)
#   MOA_GPUS=device=3 ./run.sh --gpu pytest -q -m gpu
#   MOA_GPUS=device=0,1 ./run.sh --gpu python probe.py
#
# `MOA_GPUS=all` and any selection naming more than 2 devices are rejected
# rather than silently clamped. Check `nvidia-smi` and `free -h` before starting
# a GPU job, and never run two concurrently.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE=agent_solver
DOCKERFILE=docker/Dockerfile
DO_BUILD=0
USE_GPU=0

# A loop rather than a single `case`: `--gpu` composes with `--rebuild`, so the
# leading flags can arrive in either order and in any combination.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            shift
            ;;
        --rebuild)
            DO_BUILD=1
            shift
            ;;
        --gpu)
            USE_GPU=1
            IMAGE=agent_solver_gpu
            DOCKERFILE=docker/Dockerfile.gpu
            shift
            ;;
        *)
            break
            ;;
    esac
done

gpu_args=()
if [[ "$USE_GPU" -eq 1 ]]; then
    gpus="${MOA_GPUS:-device=0}"

    # Reject `all` explicitly: this host has 8 GPUs and AGENTS.md caps the
    # project at 2. Silently clamping would hide the policy from the caller.
    if [[ "$gpus" == "all" ]]; then
        echo "run.sh: MOA_GPUS=all is not allowed (AGENTS.md caps this project at 2 GPUs)." >&2
        echo "run.sh: select explicitly, e.g. MOA_GPUS=device=0 or MOA_GPUS=device=0,1" >&2
        exit 2
    fi

    # Enforce the cap for both docker `--gpus` spellings of a quantity:
    # `device=N[,M]` (explicit IDs) and a bare integer (a count).
    device_count=""
    if [[ "$gpus" == device=* ]]; then
        device_count=$(awk -F, '{print NF}' <<<"${gpus#device=}")
    elif [[ "$gpus" =~ ^[0-9]+$ ]]; then
        device_count="$gpus"
    fi

    if [[ -n "$device_count" && "$device_count" -gt 2 ]]; then
        echo "run.sh: MOA_GPUS requests $device_count GPUs ('$gpus'); AGENTS.md caps this project at 2." >&2
        exit 2
    fi

    # The value is wrapped in *literal* double quotes, which docker requires for
    # a multi-device selection: given `--gpus device=0,1` its parser splits on
    # the comma and reads `1` as a device *count*, failing with "cannot set both
    # Count and DeviceIDs on device request". The embedded quotes suppress that
    # split. Harmless for a single device, so it is applied unconditionally.
    gpu_args=(--gpus "\"$gpus\"")
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    docker build -t "$IMAGE" -f "$DOCKERFILE" .
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
# ${gpu_args[@]+...} guards the expansion so `set -u` tolerates the empty array
# on the CPU path (bash < 4.4 treats an empty array as unset).
docker "${docker_args[@]}" \
    ${gpu_args[@]+"${gpu_args[@]}"} \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    "$@"
