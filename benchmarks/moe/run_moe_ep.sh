#!/bin/bash
# Launch the BF16 MoE EP benchmark.
#
# 1 node (4 GPUs), via torchrun:
#   bash benchmarks/moe/run_moe_ep.sh
#   bash benchmarks/moe/run_moe_ep.sh --nsys
#
# 8 nodes (EP=32), via Slurm. srun already starts one process per GPU:
#   srun -N 8 --ntasks-per-node=4 --gpus-per-node=4 --export=ALL \
#     bash benchmarks/moe/run_moe_ep.sh
#   srun -N 8 --ntasks-per-node=4 --gpus-per-node=4 --export=ALL \
#     bash benchmarks/moe/run_moe_ep.sh --nsys
#
# Extra args after --nsys are forwarded to benchmark_moe_ep.py
# (e.g. --iters 50 --mode fwd).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH="${SCRIPT_DIR}/benchmark_moe_ep.py"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NSYS=0
NSYS_OUT="${NSYS_OUT:-moe_ep}"

FORWARD_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--nsys" ]]; then
        NSYS=1
        FORWARD_ARGS+=(--profile)
    else
        FORWARD_ARGS+=("${arg}")
    fi
done

nsys_wrap() {
    local out_prefix=$1
    shift
    nsys profile \
        --output="${out_prefix}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        "$@"
}

if [[ -n "${SLURM_PROCID:-}" ]]; then
    # Already a Slurm task (one GPU). Only rank 0 records nsys to avoid 32 reports.
    if [[ "${NSYS}" == "1" && "${SLURM_PROCID}" == "0" ]]; then
        nsys_wrap "${NSYS_OUT}_rank${SLURM_PROCID}" python "${BENCH}" --gpus-per-node="${GPUS_PER_NODE}" "${FORWARD_ARGS[@]}"
    else
        python "${BENCH}" --gpus-per-node="${GPUS_PER_NODE}" "${FORWARD_ARGS[@]}"
    fi
else
    if [[ "${NSYS}" == "1" ]]; then
        nsys_wrap "${NSYS_OUT}_1node" \
            torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
            "${BENCH}" --gpus-per-node="${GPUS_PER_NODE}" "${FORWARD_ARGS[@]}"
    else
        torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
            "${BENCH}" --gpus-per-node="${GPUS_PER_NODE}" "${FORWARD_ARGS[@]}"
    fi
fi
