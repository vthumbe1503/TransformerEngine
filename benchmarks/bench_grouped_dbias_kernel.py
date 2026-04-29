#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark _grouped_dbias_kernel throughput with CUDA graph replay.

The benchmark reports an estimated memory bandwidth for the Triton kernel in
``transformer_engine.common.triton.grouped_dbias_dscales``.  It is intended for
Blackwell runs, so the default peak bandwidth is 8 TB/s and the output includes
the achieved percentage of that peak.

Examples:
    python benchmarks/bench_grouped_dbias_kernel.py --sweep
    python benchmarks/bench_grouped_dbias_kernel.py --mode both --imbalance heavy
    python benchmarks/bench_grouped_dbias_kernel.py --rows 512 --capacity-multiplier 16
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import triton

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer_engine.common.triton.grouped_dbias_dscales import (
    _grouped_dbias_kernel,
    _grouped_dbias_rowwise_kernel,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

BLOCK_M = 256
BLOCK_H = 128
N_ROW_SPLITS = 8
BLOCKS_PER_SM = 64
NUM_WARPS = 4
NUM_STAGES = 2


@dataclass(frozen=True)
class BenchCase:
    name: str
    num_groups: int
    rows_per_group: int
    hidden: int
    imbalance: str
    capacity_multiplier: float = 1.0


@dataclass(frozen=True)
class TrafficBytes:
    dy: int
    scales: int
    bias: int
    dbias_atomic: int
    dscales_atomic: int
    offsets: int

    @property
    def total(self) -> int:
        return (
            self.dy
            + self.scales
            + self.bias
            + self.dbias_atomic
            + self.dscales_atomic
            + self.offsets
        )


def generate_group_rows(num_groups: int, rows_per_group: int, imbalance: str) -> list[int]:
    """Generate exact per-group active row counts for different imbalance modes."""
    total_rows = num_groups * rows_per_group
    if total_rows < 0:
        raise ValueError("rows_per_group must be non-negative")

    if imbalance == "none":
        base = total_rows // num_groups
        rem = total_rows % num_groups
        return [base + (idx < rem) for idx in range(num_groups)]

    if imbalance == "mild":
        weights = [0.5 + (idx + 1) / num_groups for idx in range(num_groups)]
    elif imbalance == "heavy":
        weights = [1.0] * num_groups
        weights[0] = num_groups * 0.4 / 0.6
    elif imbalance == "zipf":
        weights = [1.0 / (idx + 1) for idx in range(num_groups)]
    else:
        raise ValueError(f"Unknown imbalance mode: {imbalance}")

    weight_sum = sum(weights)
    raw_rows = [total_rows * weight / weight_sum for weight in weights]
    rows = [int(math.floor(value)) for value in raw_rows]
    remainder = total_rows - sum(rows)
    order = sorted(
        range(num_groups),
        key=lambda idx: raw_rows[idx] - rows[idx],
        reverse=True,
    )
    for idx in order[:remainder]:
        rows[idx] += 1
    return rows


def make_offsets(rows: Iterable[int], device: torch.device) -> torch.Tensor:
    rows_tensor = torch.tensor(list(rows), dtype=torch.int64, device=device)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(rows_tensor, dim=0),
        ]
    )


def estimate_traffic_bytes(
    rows: list[int],
    num_groups: int,
    hidden: int,
    dtype: torch.dtype,
    has_scales: bool,
    kernel: str,
    block_m: int,
    block_h: int,
) -> TrafficBytes:
    """Estimate logical memory traffic caused by the kernel body.

    The estimate intentionally counts repeated scale and bias loads across
    column blocks / row splits, plus fp32 atomic read-modify-write traffic.
    Actual DRAM bytes may differ because of cache hits and the GPU's atomic
    implementation, but this gives a stable apples-to-apples throughput number.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    col_blocks = triton.cdiv(hidden, block_h)
    active_rows = sum(rows)

    if kernel == "old":
        dbias_contributors = num_groups * N_ROW_SPLITS
    elif kernel == "rowwise":
        dbias_contributors = sum(triton.cdiv(row_count, block_m) for row_count in rows)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    dy_bytes = active_rows * hidden * elem_size
    dbias_atomic_bytes = dbias_contributors * hidden * 2 * 4
    offsets_bytes = max(1, dbias_contributors) * col_blocks * 2 * 8

    scales_bytes = 0
    bias_bytes = 0
    dscales_atomic_bytes = 0
    if has_scales:
        scales_bytes = active_rows * col_blocks * 4
        dscales_atomic_bytes = active_rows * col_blocks * 2 * 4
        bias_bytes = dbias_contributors * hidden * elem_size

    return TrafficBytes(
        dy=dy_bytes,
        scales=scales_bytes,
        bias=bias_bytes,
        dbias_atomic=dbias_atomic_bytes,
        dscales_atomic=dscales_atomic_bytes,
        offsets=offsets_bytes,
    )


def launch_grouped_dbias_kernel(
    dy: torch.Tensor,
    offsets: torch.Tensor,
    dbias: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    dscales: torch.Tensor,
    has_scales: bool,
    kernel: str,
    block_m: int,
    blocks_per_sm: int,
    block_h: int,
) -> None:
    hidden = dy.shape[1]
    num_groups = dbias.shape[0]
    col_blocks = triton.cdiv(hidden, block_h)

    if kernel == "old":
        grid = (num_groups, N_ROW_SPLITS, col_blocks)
        _grouped_dbias_kernel[grid](
            dy,
            dbias,
            offsets,
            scales,
            bias,
            dscales,
            hidden,
            HAS_SCALES=has_scales,
            N_ROW_SPLITS=N_ROW_SPLITS,
            BLOCK_M=block_m,
            BLOCK_H=block_h,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    elif kernel == "rowwise":
        num_sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
        max_row_workers = triton.cdiv(dy.shape[0], block_m)
        row_workers = min(
            max_row_workers,
            max(1, num_sms * blocks_per_sm // col_blocks),
        )
        grid = (max(1, row_workers), col_blocks)
        _grouped_dbias_rowwise_kernel[grid](
            dy,
            dbias,
            offsets,
            scales,
            bias,
            dscales,
            hidden,
            num_groups,
            HAS_SCALES=has_scales,
            BLOCK_M=block_m,
            BLOCK_H=block_h,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def resolve_block_h(kernel: str, hidden: int, block_h: int) -> int:
    if block_h > 0:
        return block_h
    return BLOCK_H


def bench_cuda_graph(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    replays_per_iter: int,
) -> float:
    """Return average graph replay time in milliseconds."""
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        for _ in range(replays_per_iter):
            graph.replay()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / (iters * replays_per_iter)


def run_case(
    case: BenchCase,
    dtype: torch.dtype,
    has_scales: bool,
    warmup: int,
    iters: int,
    replays_per_iter: int,
    peak_bandwidth_tb_s: float,
    device: torch.device,
    kernel: str,
    block_m: int,
    blocks_per_sm: int,
    block_h: int,
) -> None:
    rows = generate_group_rows(case.num_groups, case.rows_per_group, case.imbalance)
    active_rows = sum(rows)
    capacity_rows = max(active_rows, math.ceil(active_rows * case.capacity_multiplier))
    offsets = make_offsets(rows, device)

    dy = torch.randn((capacity_rows, case.hidden), dtype=dtype, device=device)
    dbias = torch.empty((case.num_groups, case.hidden), dtype=torch.float32, device=device)

    if has_scales:
        scales = torch.randn((capacity_rows,), dtype=torch.float32, device=device)
        bias = torch.randn((case.num_groups, case.hidden), dtype=dtype, device=device)
        dscales = torch.empty((capacity_rows,), dtype=torch.float32, device=device)
    else:
        # Triton still requires valid pointers for the unused arguments.
        scales = dy
        bias = dy
        dscales = dy

    def kernel_call() -> None:
        launch_grouped_dbias_kernel(
            dy,
            offsets,
            dbias,
            scales,
            bias,
            dscales,
            has_scales,
            kernel,
            block_m,
            blocks_per_sm,
            block_h,
        )

    elapsed_ms = bench_cuda_graph(kernel_call, warmup, iters, replays_per_iter)
    traffic = estimate_traffic_bytes(
        rows=rows,
        num_groups=case.num_groups,
        hidden=case.hidden,
        dtype=dtype,
        has_scales=has_scales,
        kernel=kernel,
        block_m=block_m,
        block_h=block_h,
    )
    bandwidth_tb_s = traffic.total / (elapsed_ms * 1e-3) / 1e12
    pct_peak = 100.0 * bandwidth_tb_s / peak_bandwidth_tb_s
    dy_only_tb_s = traffic.dy / (elapsed_ms * 1e-3) / 1e12

    min_rows = min(rows) if rows else 0
    max_rows = max(rows) if rows else 0
    row_ratio = max_rows / max(min_rows, 1)
    fill_pct = 100.0 * active_rows / max(capacity_rows, 1)

    print(
        f"{case.name:>22} "
        f"{kernel:>8} "
        f"{block_m:>4} "
        f"{block_h:>4} "
        f"{blocks_per_sm:>4} "
        f"{case.num_groups:>5} "
        f"{active_rows:>9} "
        f"{capacity_rows:>9} "
        f"{fill_pct:>6.1f} "
        f"{case.hidden:>6} "
        f"{'Y' if has_scales else 'N':>5} "
        f"{case.imbalance:>10} "
        f"{min_rows:>7} "
        f"{max_rows:>7} "
        f"{row_ratio:>6.1f} "
        f"{traffic.total / 1e6:>10.1f} "
        f"{elapsed_ms:>9.4f} "
        f"{bandwidth_tb_s:>8.3f} "
        f"{pct_peak:>7.1f} "
        f"{dy_only_tb_s:>9.3f}"
    )


def sweep_cases() -> list[BenchCase]:
    return [
        # Uniform baselines from bench_grouped_bias.py.
        BenchCase("uniform_1x4096", 1, 4096, 4096, "none"),
        BenchCase("uniform_1x32768", 1, 32768, 4096, "none"),
        BenchCase("uniform_1x65536", 1, 65536, 4096, "none"),
        BenchCase("uniform_4x4096", 4, 4096, 4096, "none"),
        BenchCase("uniform_8x4096", 8, 4096, 4096, "none"),
        BenchCase("uniform_8x8192", 8, 8192, 4096, "none"),
        BenchCase("uniform_16x4096", 16, 4096, 4096, "none"),
        BenchCase("uniform_32x2048", 32, 2048, 4096, "none"),
        BenchCase("wide_8x4096", 8, 4096, 12288, "none"),
        BenchCase("many_64x1024", 64, 1024, 4096, "none"),
        # Real workload shape from bench_grouped_bias.py.
        BenchCase("real_16x6144x2880", 16, 6144, 2880, "none"),
        # MoE-like imbalance cases from bench_grouped_bias.py.
        BenchCase("mild_8x4096", 8, 4096, 4096, "mild"),
        BenchCase("mild_16x4096", 16, 4096, 4096, "mild"),
        BenchCase("mild_64x1024", 64, 1024, 4096, "mild"),
        BenchCase("heavy_8x4096", 8, 4096, 4096, "heavy"),
        BenchCase("heavy_16x4096", 16, 4096, 4096, "heavy"),
        BenchCase("zipf_8x4096", 8, 4096, 4096, "zipf"),
        BenchCase("zipf_16x4096", 16, 4096, 4096, "zipf"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark _grouped_dbias_kernel bandwidth")
    parser.add_argument("--num-groups", type=int, default=16, help="Number of groups")
    parser.add_argument("--rows", type=int, default=6144, help="Average active rows per group")
    parser.add_argument("--hidden", type=int, default=2880, help="Hidden dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=DTYPE_MAP.keys(),
        help="dy/bias dtype",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dbias",
        choices=["dbias", "fused", "both"],
        help="'dbias' benchmarks HAS_SCALES=False; 'fused' includes scales, bias, and dscales",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rowwise",
        choices=["old", "rowwise", "both"],
        help="Triton kernel scheduler to benchmark",
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        default="none",
        choices=["none", "mild", "heavy", "zipf"],
        help="Per-group active row distribution",
    )
    parser.add_argument(
        "--imbalance-sweep",
        action="store_true",
        help="Run the selected size with none, mild, heavy, and zipf imbalance",
    )
    parser.add_argument(
        "--capacity-multiplier",
        type=float,
        default=1.0,
        help="Allocate this many times more rows than sum(offsets) to test underfilled tensors",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        default=BLOCK_M,
        help="Rows per Triton tile/chunk",
    )
    parser.add_argument(
        "--block-m-sweep",
        action="store_true",
        help="Sweep BLOCK_M over 64,128,256,512",
    )
    parser.add_argument(
        "--blocks-per-sm",
        type=int,
        default=BLOCKS_PER_SM,
        help="Row-worker blocks per SM for the rowwise kernel",
    )
    parser.add_argument(
        "--block-h",
        type=int,
        default=0,
        help="Columns per Triton program; 0 uses 128",
    )
    parser.add_argument(
        "--block-h-sweep",
        action="store_true",
        help="Override auto BLOCK_H and sweep over 64,128,256,512",
    )
    parser.add_argument(
        "--blocks-per-sm-sweep",
        action="store_true",
        help="Sweep rowwise-kernel row-worker blocks per SM over 1,2,4,8,16,32",
    )
    parser.add_argument("--warmup", type=int, default=50, help="CUDA graph replay warmup count")
    parser.add_argument("--iters", type=int, default=100, help="Timed outer iterations")
    parser.add_argument(
        "--replays-per-iter",
        type=int,
        default=10,
        help="Graph replays inside one event timing window iteration",
    )
    parser.add_argument(
        "--peak-bandwidth-tb-s",
        type=float,
        default=8.0,
        help="Peak HBM bandwidth used for percent-of-peak reporting",
    )
    parser.add_argument("--sweep", action="store_true", help="Run a built-in sweep of cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.capacity_multiplier < 1.0:
        raise ValueError("--capacity-multiplier must be >= 1")
    if args.blocks_per_sm < 1:
        raise ValueError("--blocks-per-sm must be >= 1")
    if args.block_m < 1:
        raise ValueError("--block-m must be >= 1")
    if args.block_h < 0:
        raise ValueError("--block-h must be >= 0")

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")
    modes = [False, True] if args.mode == "both" else [args.mode == "fused"]
    kernels = ["old", "rowwise"] if args.kernel == "both" else [args.kernel]
    block_m_values = [64, 128, 256, 512] if args.block_m_sweep else [args.block_m]
    blocks_per_sm_values = [1, 2, 4, 8, 16, 32] if args.blocks_per_sm_sweep else [
        args.blocks_per_sm
    ]
    block_h_values = [64, 128, 256, 512] if args.block_h_sweep else [args.block_h]
    if args.sweep:
        cases = sweep_cases()
    elif args.imbalance_sweep:
        cases = [
            BenchCase(
                f"{imbalance}_{args.num_groups}x{args.rows}x{args.hidden}",
                args.num_groups,
                args.rows,
                args.hidden,
                imbalance,
                args.capacity_multiplier,
            )
            for imbalance in ("none", "mild", "heavy", "zipf")
        ]
    else:
        cases = [
            BenchCase(
                "custom",
                args.num_groups,
                args.rows,
                args.hidden,
                args.imbalance,
                args.capacity_multiplier,
            )
        ]

    print(f"Peak HBM bandwidth target: {args.peak_bandwidth_tb_s:.2f} TB/s")
    print(
        f"Kernel constants: default_BLOCK_M={BLOCK_M}, default_BLOCK_H={BLOCK_H}, "
        "rowwise uses column-block grid, "
        f"N_ROW_SPLITS={N_ROW_SPLITS}, "
        f"default_BLOCKS_PER_SM={BLOCKS_PER_SM}, num_warps={NUM_WARPS}, num_stages={NUM_STAGES}"
    )
    print(
        f"Timing: CUDA graph replay, {args.warmup} warmups, "
        f"{args.iters} x {args.replays_per_iter} timed replays"
    )
    print(
        "Traffic model: dy + repeated scale/bias loads + fp32 atomic RMWs "
        "+ offset loads; dy_TB/s is shown separately."
    )
    print()
    print(
        f"{'case':>22} {'kernel':>8} {'BM':>4} {'BH':>4} {'b/sm':>4} "
        f"{'grps':>5} {'active':>9} "
        f"{'capacity':>9} {'fill%':>6} "
        f"{'hidden':>6} {'scale':>5} {'imbal':>10} {'min_r':>7} {'max_r':>7} "
        f"{'ratio':>6} {'est_MB':>10} {'ms':>9} {'TB/s':>8} {'%8TB/s':>7} {'dy_TB/s':>9}"
    )
    print("-" * 182)

    for case in cases:
        for kernel in kernels:
            kernel_blocks_per_sm_values = blocks_per_sm_values if kernel == "rowwise" else [
                args.blocks_per_sm
            ]
            for block_m in block_m_values:
                for block_h in block_h_values:
                    for blocks_per_sm in kernel_blocks_per_sm_values:
                        for has_scales in modes:
                            actual_block_h = resolve_block_h(kernel, case.hidden, block_h)
                            run_case(
                                case=case,
                                dtype=dtype,
                                has_scales=has_scales,
                                warmup=args.warmup,
                                iters=args.iters,
                                replays_per_iter=args.replays_per_iter,
                                peak_bandwidth_tb_s=args.peak_bandwidth_tb_s,
                                device=device,
                                kernel=kernel,
                                block_m=block_m,
                                blocks_per_sm=blocks_per_sm,
                                block_h=actual_block_h,
                            )


if __name__ == "__main__":
    main()
