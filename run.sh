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

    # Fallback for a host whose NVML enumeration is broken.
    #
    # `--gpus` goes through nvidia-container-cli, whose prestart hook calls
    # nvmlInit and enumerates *every* GPU on the box before it decides which
    # ones to expose. One GPU in a fault state therefore fails the hook for all
    # of them -- "nvidia-container-cli: detection error: nvml error: unknown
    # error" -- and no GPU container starts at all, however healthy the device
    # actually requested is. Recovering the faulted device needs a GPU reset or
    # a reboot, i.e. root on a shared host, so it is not something this script
    # can or should do.
    #
    # What the hook does is mechanical: bind the requested /dev/nvidia* nodes
    # and the host's userspace driver libraries into the container. Doing it
    # explicitly skips the enumeration entirely. It is strictly narrower than
    # `--gpus` -- only the devices named by MOA_GPUS are visible -- and it
    # touches no host state.
    #
    # MOA_GPU_PASSTHROUGH=1 forces this path, =0 forbids it; unset auto-detects.
    passthrough="${MOA_GPU_PASSTHROUGH:-auto}"
    if [[ "$passthrough" == auto ]]; then
        if nvidia-container-cli info >/dev/null 2>&1; then
            passthrough=0
        else
            passthrough=1
            echo "run.sh: nvidia-container-cli cannot enumerate this host's GPUs;" >&2
            echo "run.sh: falling back to explicit device passthrough for '$gpus'." >&2
        fi
    fi

    if [[ "$passthrough" == 1 ]]; then
        if [[ "$gpus" != device=* ]]; then
            echo "run.sh: device passthrough needs explicit ids, e.g. MOA_GPUS=device=0 (got '$gpus')." >&2
            exit 2
        fi

        gpu_args=()
        IFS=, read -r -a _moa_devs <<<"${gpus#device=}"
        for dev in "${_moa_devs[@]}"; do
            if [[ ! -e "/dev/nvidia${dev}" ]]; then
                echo "run.sh: /dev/nvidia${dev} does not exist." >&2
                exit 2
            fi
            gpu_args+=(--device "/dev/nvidia${dev}")
        done
        for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
            [[ -e "$dev" ]] && gpu_args+=(--device "$dev")
        done

        # Mount each driver library straight onto its SONAME. The container runs
        # as the invoking user, so it cannot create the symlinks or run ldconfig
        # that the real hook would; binding onto the linker-visible name is the
        # equivalent that works unprivileged.
        libdir=/usr/lib/x86_64-linux-gnu
        driver_version=$(cat /sys/module/nvidia/version 2>/dev/null || true)
        if [[ -z "$driver_version" ]]; then
            echo "run.sh: cannot read the NVIDIA driver version from /sys/module/nvidia/version." >&2
            exit 2
        fi
        for pair in \
            "libcuda.so:libcuda.so.1" \
            "libnvidia-ml.so:libnvidia-ml.so.1" \
            "libnvidia-ptxjitcompiler.so:libnvidia-ptxjitcompiler.so.1" \
            "libnvidia-nvvm.so:libnvidia-nvvm.so.4" \
            "libnvidia-gpucomp.so:libnvidia-gpucomp.so.1" \
            "libnvidia-allocator.so:libnvidia-allocator.so.1"
        do
            src="${libdir}/${pair%%:*}.${driver_version}"
            dst="${libdir}/${pair##*:}"
            [[ -e "$src" ]] && gpu_args+=(-v "${src}:${dst}:ro")
        done
        [[ -x /usr/bin/nvidia-smi ]] && gpu_args+=(-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro)
        gpu_args+=(-e "LD_LIBRARY_PATH=${libdir}")
    fi
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
