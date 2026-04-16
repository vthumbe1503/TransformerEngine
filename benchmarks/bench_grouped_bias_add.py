#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark for grouped_bias_add_kernel throughput.

Measures effective memory bandwidth (TB/s) with and without CUDA graphs,
and compares against the GPU's theoretical peak HBM bandwidth.

Usage:
    python benchmarks/bench_grouped_bias_add.py [--use-scale] [--num-tensors N]
                                                 [--rows M] [--hidden H]
                                                 [--dtype {bf16,fp16,fp32}]
                                                 [--warmup W] [--iters I]
    python benchmarks/bench_grouped_bias_add.py --ncu [--num-tensors 8] [--rows 4096]
"""

import argparse
import os
import subprocess
import sys
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def get_gpu_peak_bandwidth_tb_s() -> float:
    """Query nvidia-smi for memory clock + bus width and compute theoretical peak."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,clocks.max.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        line = out.strip().split("\n")[0]
        mem_total_mib, mem_clock_mhz = [int(x.strip()) for x in line.split(",")]

        # B100 SXM 192GB -> 8 TB/s;  detect by memory size
        if mem_total_mib > 170_000:
            return 8.0
        elif mem_total_mib > 75_000:
            return 3.35  # H100 SXM ~3.35 TB/s
        elif mem_total_mib > 35_000:
            return 2.0   # A100 80GB
        else:
            return 1.555 # A100 40GB
    except Exception:
        return 8.0  # default to B100 SXM


def _generate_imbalanced_rows(num_tensors: int, avg_rows: int, imbalance: str) -> list:
    """Generate per-expert row counts that simulate realistic load imbalance.

    imbalance modes:
      'none'    - all experts get the same number of rows
      'mild'    - MoE-like mild skew (~2x ratio between largest/smallest)
      'heavy'   - heavy skew (~8-10x ratio, simulates popular-expert hotspot)
      'zipf'    - Zipf-law distribution (rank-proportional)
    All modes preserve total_rows = num_tensors * avg_rows (approximately),
    and each expert gets at least 128 rows (alignment-friendly).
    """
    import random
    random.seed(42)
    total = num_tensors * avg_rows

    if imbalance == "none":
        return [avg_rows] * num_tensors

    if imbalance == "mild":
        # Dirichlet-like: sample weights ~ Uniform then rescale
        weights = [random.uniform(0.5, 1.5) for _ in range(num_tensors)]
    elif imbalance == "heavy":
        # One hot expert gets ~40% of tokens, rest share the remainder
        weights = [1.0] * num_tensors
        weights[0] = num_tensors * 0.4 / 0.6
    elif imbalance == "zipf":
        # Zipf: weight_i ~ 1 / (rank+1)
        weights = [1.0 / (i + 1) for i in range(num_tensors)]
    else:
        raise ValueError(f"Unknown imbalance mode: {imbalance}")

    wsum = sum(weights)
    # Round to multiples of 4 (kVec alignment) with minimum of 128 rows
    row_list = [max(128, round(w / wsum * total / 4) * 4) for w in weights]
    # Adjust last expert so total is exact
    row_list[-1] = max(128, total - sum(row_list[:-1]))
    return row_list


def make_grouped_output(
    num_tensors: int,
    rows_per_tensor: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    imbalance: str = "none",
) -> GroupedTensor:
    """Create a GroupedTensor for the output (num_tensors x rows x hidden_size)."""
    row_list = _generate_imbalanced_rows(num_tensors, rows_per_tensor, imbalance)
    total_rows = sum(row_list)

    if imbalance != "none":
        first_dims = torch.tensor(row_list, dtype=torch.int64, device=device)
    else:
        first_dims = None

    total_elements = total_rows * hidden_size
    data = torch.randn(total_elements, dtype=dtype, device=device)
    logical_shape = (total_rows, hidden_size)
    shapes = [(r, hidden_size) for r in row_list]

    tensor_offsets = None
    if first_dims is not None:
        tensor_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(first_dims * hidden_size, dim=0),
        ])

    return GroupedTensor(
        shape=logical_shape,
        dtype=dtype,
        num_tensors=num_tensors,
        shapes=shapes,
        data=data,
        first_dims=first_dims,
        tensor_offsets=tensor_offsets,
    ), total_rows


def make_grouped_bias(
    num_tensors: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> GroupedTensor:
    """Create a GroupedTensor for bias (num_tensors x 1 x hidden_size)."""
    total_elements = num_tensors * hidden_size
    data = torch.randn(total_elements, dtype=dtype, device=device)
    logical_shape = (num_tensors, hidden_size)
    shapes = [(1, hidden_size)] * num_tensors

    return GroupedTensor(
        shape=logical_shape,
        dtype=dtype,
        num_tensors=num_tensors,
        shapes=shapes,
        data=data,
    )


def compute_bandwidth_tb_s(
    total_rows: int,
    hidden_size: int,
    num_tensors: int,
    dtype: torch.dtype,
    use_scale: bool,
    elapsed_ms: float,
) -> float:
    """Compute effective memory bandwidth in TB/s.

    Traffic:
      - Read output:  total_rows * hidden_size * elem_size
      - Write output: total_rows * hidden_size * elem_size
      - Read bias:    num_tensors * hidden_size * elem_size  (broadcast across rows)
      - Read scale:   total_rows * 4 bytes  (if use_scale, fp32)
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    output_bytes = total_rows * hidden_size * elem_size
    bias_bytes = num_tensors * hidden_size * elem_size
    scale_bytes = total_rows * 4 if use_scale else 0
    total_bytes = 2 * output_bytes + bias_bytes + scale_bytes
    elapsed_s = elapsed_ms / 1000.0
    return (total_bytes / elapsed_s) / 1e12


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_no_graph(
    output_gt: GroupedTensor,
    bias_gt: GroupedTensor,
    bias_scale: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    """Benchmark without CUDA graphs. Returns median time in ms."""
    for _ in range(warmup):
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]


def bench_with_graph(
    output_gt: GroupedTensor,
    bias_gt: GroupedTensor,
    bias_scale: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    """Benchmark with CUDA graph capture. Returns median time in ms."""
    # Warmup before capture
    for _ in range(3):
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)

    # Warmup replays
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    # Timed replays
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        graph.replay()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ncu_profile(args):
    """Re-launch this script under ncu to collect kernel metrics."""
    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")

    output_gt, total_rows = make_grouped_output(
        args.num_tensors, args.rows, args.hidden, dtype, device, args.imbalance
    )
    bias_gt = make_grouped_bias(args.num_tensors, args.hidden, dtype, device)
    if args.use_scale:
        bias_scale = torch.randn(total_rows, dtype=torch.float32, device=device)
    else:
        bias_scale = torch.empty(0, dtype=torch.float32, device=device)

    # Warmup to JIT-compile and stabilise
    for _ in range(5):
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()

    # Profile with CUPTI range
    torch.cuda.cudart().cudaProfilerStart()
    tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("Profiled kernel invocation complete.  Run this script under ncu:")
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = (2 * total_rows * args.hidden * elem_size
                   + args.num_tensors * args.hidden * elem_size
                   + (total_rows * 4 if args.use_scale else 0))
    print(f"  Config: {args.num_tensors} tensors, {args.rows} rows, {args.hidden} hidden, "
          f"scale={'Y' if args.use_scale else 'N'}, imbalance={args.imbalance}")
    print(f"  Expected traffic: {total_bytes / 1e6:.2f} MB")
    print()
    print("Example ncu command (run from shell, not via this script):")
    ncu_cmd = (
        f"ncu --set full --target-processes all "
        f"--kernel-name grouped_bias_add_kernel "
        f"-o /tmp/grouped_bias_add_profile "
        f"python {os.path.abspath(__file__)} "
        f"--ncu-target "
        f"--num-tensors {args.num_tensors} --rows {args.rows} --hidden {args.hidden} "
        f"--dtype {args.dtype} --imbalance {args.imbalance}"
    )
    if args.use_scale:
        ncu_cmd += " --use-scale"
    print(f"  {ncu_cmd}")
    print()
    print("Key metrics to look at in the ncu output:")
    print("  - sm__throughput.avg.pct_of_peak_sustained_elapsed  (SM utilisation)")
    print("  - dram__throughput.avg.pct_of_peak_sustained_elapsed (HBM utilisation)")
    print("  - l1tex__throughput.avg.pct_of_peak_sustained_elapsed (L1 utilisation)")
    print("  - lts__throughput.avg.pct_of_peak_sustained_elapsed (L2 utilisation)")
    print("  - sm__warps_active.avg.pct_of_peak_sustained_elapsed (occupancy)")
    print("  - dram__bytes.sum (total DRAM bytes transferred)")
    print("  - launch__registers_per_thread (registers per thread)")
    print("  - launch__block_size (block size)")
    print("  - launch__grid_size (grid size)")


def run_ncu_target(args):
    """Entry point when running under ncu --target-processes."""
    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")

    output_gt, total_rows = make_grouped_output(
        args.num_tensors, args.rows, args.hidden, dtype, device, args.imbalance
    )
    bias_gt = make_grouped_bias(args.num_tensors, args.hidden, dtype, device)
    if args.use_scale:
        bias_scale = torch.randn(total_rows, dtype=torch.float32, device=device)
    else:
        bias_scale = torch.empty(0, dtype=torch.float32, device=device)

    # Warmup outside profiler range
    for _ in range(3):
        tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()

    # Single invocation for ncu to capture
    tex.te_grouped_bias_add(output_gt, bias_gt, bias_scale)
    torch.cuda.synchronize()
    print("ncu target invocation done.", file=sys.stderr)


def run_ncu_inline(args):
    """Launch ncu as a subprocess and parse key metrics from its CSV output.

    ncu --csv emits one row per metric with columns like:
      "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
    We parse "Metric Name" -> "Metric Value" into a dict.
    """
    dtype = DTYPE_MAP[args.dtype]

    target_cmd = [
        sys.executable, os.path.abspath(__file__),
        "--ncu-target",
        "--num-tensors", str(args.num_tensors),
        "--rows", str(args.rows),
        "--hidden", str(args.hidden),
        "--dtype", args.dtype,
        "--imbalance", args.imbalance,
    ]
    if args.use_scale:
        target_cmd.append("--use-scale")

    ncu_metrics = [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
        "launch__registers_per_thread",
        "launch__block_size",
        "launch__grid_size",
        "sm__sass_thread_inst_executed_op_memory_128b.sum",
        "sm__sass_thread_inst_executed_op_memory_64b.sum",
        "sm__sass_thread_inst_executed_op_memory_32b.sum",
        "gpu__time_duration.sum",
    ]

    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", "grouped_bias_add_kernel",
        "--launch-skip", "3",
        "--launch-count", "1",
        "--metrics", ",".join(ncu_metrics),
        "--csv",
    ] + target_cmd

    print(f"Config: {args.num_tensors} tensors, {args.rows} rows, "
          f"{args.hidden} hidden, scale={'Y' if args.use_scale else 'N'}, "
          f"imbalance={args.imbalance}, dtype={args.dtype}")
    print(f"Running: {' '.join(ncu_cmd)}")
    print()

    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        print("ERROR: 'ncu' not found. Install NVIDIA Nsight Compute or add it to PATH.")
        print("  Typical location: /usr/local/cuda/bin/ncu")
        return
    except subprocess.TimeoutExpired:
        print("ERROR: ncu timed out after 300s")
        return

    if result.returncode != 0:
        print("ncu stderr:", result.stderr[:2000])
        print("ncu stdout:", result.stdout[:2000])
        print("ncu exited with code", result.returncode)
        return

    # Parse CSV: ncu outputs one row per metric.
    # Filter out banner lines (start with "==") and blank lines.
    import csv
    import io

    raw = result.stdout
    # ncu CSV lines always start with a quoted field like "ID" or "0".
    # Skip banners (==...) and any stray print() output from the target process.
    csv_lines = [l for l in raw.split("\n") if l.strip() and l.lstrip().startswith('"')]
    if len(csv_lines) < 2:
        print("Could not parse ncu CSV output. Raw stdout:")
        print(raw[:5000])
        return

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))

    # Find the column names for metric name/value/unit.
    # ncu uses headers like: "Metric Name", "Metric Value", "Metric Unit"
    # but exact names can vary. We search case-insensitively.
    metrics = {}  # metric_name -> (value_str, unit_str)
    for row in reader:
        # Normalise keys: strip quotes and whitespace
        norm = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items() if k}

        name = norm.get("Metric Name", norm.get("metric_name", ""))
        value = norm.get("Metric Value", norm.get("metric_value", ""))
        unit = norm.get("Metric Unit", norm.get("metric_unit", ""))
        if name:
            metrics[name] = (value, unit)

    if not metrics:
        print("No metrics parsed. Dumping raw ncu CSV output for debugging:")
        print("-" * 70)
        for line in csv_lines[:30]:
            print(line)
        print("-" * 70)
        print(f"(total {len(csv_lines)} CSV lines)")
        return

    # Compute expected traffic for comparison
    elem_size = torch.tensor([], dtype=DTYPE_MAP[args.dtype]).element_size()
    total_rows_val = args.num_tensors * args.rows
    expected_bytes = (2 * total_rows_val * args.hidden * elem_size
                      + args.num_tensors * args.hidden * elem_size
                      + (total_rows_val * 4 if args.use_scale else 0))

    print("=" * 80)
    print("Nsight Compute Kernel Metrics")
    print("=" * 80)
    for metric in ncu_metrics:
        val, unit = metrics.get(metric, ("N/A", ""))
        unit_str = f" {unit}" if unit else ""
        print(f"  {metric:<60s} = {val}{unit_str}")

    # Also print any other metrics ncu returned that we didn't explicitly ask for
    extra = set(metrics.keys()) - set(ncu_metrics)
    if extra:
        print()
        print("  Additional metrics returned by ncu:")
        for m in sorted(extra):
            val, unit = metrics[m]
            unit_str = f" {unit}" if unit else ""
            print(f"  {m:<60s} = {val}{unit_str}")

    print()
    print(f"Expected effective traffic:  {expected_bytes / 1e6:.2f} MB")
    dram_val, _ = metrics.get("dram__bytes.sum", (None, ""))
    if dram_val and dram_val != "N/A":
        try:
            dram_total = float(dram_val.replace(",", ""))
            print(f"Actual DRAM bytes (ncu):     {dram_total / 1e6:.2f} MB")
            if expected_bytes > 0:
                print(f"Traffic amplification:       {dram_total / expected_bytes:.2f}x")
        except ValueError:
            print(f"Actual DRAM bytes (ncu):     {dram_val} (could not parse)")
    print("=" * 80)


@triton.jit
def _triton_add_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, a + b, mask=mask)


@triton.jit
def _triton_copy_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, b, mask=mask)


@triton.jit
def _triton_add_row_bcast_kernel(
    a_ptr, bias_ptr, cols, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    col_idx = offsets % cols
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + col_idx, mask=mask)
    tl.store(a_ptr + offsets, a + b, mask=mask)


@triton.jit
def _triton_add_outofplace_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


_TRITON_BS = 8192


def _triton_add(a, b):
    n = a.numel()
    grid = ((n + _TRITON_BS - 1) // _TRITON_BS,)
    _triton_add_kernel[grid](a, b, n, BLOCK_SIZE=_TRITON_BS)


def _triton_copy(a, b):
    n = a.numel()
    grid = ((n + _TRITON_BS - 1) // _TRITON_BS,)
    _triton_copy_kernel[grid](a, b, n, BLOCK_SIZE=_TRITON_BS)


def _triton_add_row_bcast(a, bias, rows, cols):
    n = a.numel()
    grid = ((n + _TRITON_BS - 1) // _TRITON_BS,)
    _triton_add_row_bcast_kernel[grid](a, bias, cols, n, BLOCK_SIZE=_TRITON_BS)


def _triton_add_oop(a, b, c):
    n = a.numel()
    grid = ((n + _TRITON_BS - 1) // _TRITON_BS,)
    _triton_add_outofplace_kernel[grid](a, b, c, n, BLOCK_SIZE=_TRITON_BS)


def _baseline_bench_graph(fn, warmup, iters):
    """Benchmark a callable via CUDA graph replay. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return times[len(times) // 2]


def _bench_event_batched(fn, warmup, iters):
    """Benchmark using batched event timing (matches vecadd benchmark methodology)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_baseline(args):
    """Benchmark simple PyTorch/Triton ops to establish achievable HBM bandwidth."""
    device = torch.device("cuda")
    peak_bw = get_gpu_peak_bandwidth_tb_s()
    dtype = DTYPE_MAP[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    warmup, iters = args.warmup, args.iters

    sizes = [
        (4096, 4096),
        (32768, 4096),
        (65536, 4096),
        (65536, 12288),
        (98304, 2880),
        (98304, 4096),
    ]

    print(f"GPU peak HBM bandwidth: {peak_bw:.2f} TB/s")
    print(f"dtype: {args.dtype} ({elem_size} bytes), Triton BLOCK_SIZE: {_TRITON_BS}")
    print(f"Warmup: {warmup}, Iters: {iters}")
    print(f"Timing: batched events (matching vecadd benchmark)")
    print("=" * 115)
    print(f"{'rows':>7} {'cols':>6} {'MB/arr':>8} {'op':>18} "
          f"{'eff_bytes':>10} {'ms':>10} {'TB/s':>8} {'%peak':>7}")
    print("-" * 115)

    for rows, cols in sizes:
        n = rows * cols
        a = torch.randn(n, dtype=dtype, device=device)
        b = torch.randn(n, dtype=dtype, device=device)
        c = torch.empty(n, dtype=dtype, device=device)
        b_row = torch.randn(cols, dtype=dtype, device=device)
        mb_per_arr = n * elem_size / 1e6

        ops = [
            ("pt:mul_(2.0)",    lambda: a.mul_(2.0),                       2 * n * elem_size),
            ("pt:copy_",        lambda: a.copy_(b),                        2 * n * elem_size),
            ("pt:add_",         lambda: a.add_(b),                         3 * n * elem_size),
            ("tt:copy",         lambda: _triton_copy(a, b),                2 * n * elem_size),
            ("tt:add_inplace",  lambda: _triton_add(a, b),                 3 * n * elem_size),
            ("tt:add_oop",      lambda: _triton_add_oop(a, b, c),          3 * n * elem_size),
            ("tt:add_bcast",    lambda: _triton_add_row_bcast(a, b_row, rows, cols),
                                                                           2 * n * elem_size + cols * elem_size),
        ]

        for name, fn, eff_bytes in ops:
            t = _bench_event_batched(fn, warmup, iters)
            bw = eff_bytes / (t * 1e-3) / 1e12
            pct = 100.0 * bw / peak_bw
            print(f"{rows:>7} {cols:>6} {mb_per_arr:>8.1f} {name:>18} "
                  f"{eff_bytes/1e6:>9.1f}M {t:>10.4f} {bw:>8.3f} {pct:>6.1f}%")
        print()

    print("=" * 115)
    print("pt:* = PyTorch native,  tt:* = Triton (BLOCK_SIZE={})".format(_TRITON_BS))
    print("mul_(2.0):    read A + write A                   = 2x  (pure RMW baseline)")
    print("copy_:        read B + write A                   = 2x  (two arrays)")
    print("add_inplace:  read A + read B + write A          = 3x  (in-place, like a+=b)")
    print("add_oop:      read A + read B + write C          = 3x  (out-of-place, like vecadd benchmark)")
    print("add_bcast:    read A + broadcast row + write A   = 2x + row  (closest to our kernel)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped_bias_add_kernel")
    parser.add_argument("--use-scale", action="store_true", help="Enable per-row scaling")
    parser.add_argument("--num-tensors", type=int, default=8, help="Number of tensors in group")
    parser.add_argument("--rows", type=int, default=4096, help="Rows per tensor (M)")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size (N)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=DTYPE_MAP.keys())
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations")
    parser.add_argument("--imbalance", type=str, default="none",
                        choices=["none", "mild", "heavy", "zipf"],
                        help="Load imbalance mode for per-expert row counts")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over multiple configurations")
    parser.add_argument("--ncu", action="store_true",
                        help="Run Nsight Compute profiling on a single config")
    parser.add_argument("--ncu-target", action="store_true",
                        help=argparse.SUPPRESS)  # internal: entry point when running under ncu
    parser.add_argument("--baseline", action="store_true",
                        help="Benchmark simple PyTorch ops (copy_, add_) to establish peak achievable BW")
    args = parser.parse_args()

    if args.ncu_target:
        run_ncu_target(args)
        return

    if args.ncu:
        run_ncu_inline(args)
        return

    if args.baseline:
        run_baseline(args)
        return

    device = torch.device("cuda")
    peak_bw = get_gpu_peak_bandwidth_tb_s()

    print(f"GPU peak HBM bandwidth: {peak_bw:.2f} TB/s")
    print(f"{'='*90}")

    if args.sweep:
        configs = [
            # (num_tensors, rows, hidden, use_scale, imbalance)
            # --- Uniform baselines ---
            (1,   4096, 4096,  False, "none"),
            (1,   4096, 4096,  True,  "none"),
            (1,  32768, 4096,  False, "none"),
            (1,  32768, 4096,  True,  "none"),
            (1,  65536, 4096,  False, "none"),
            (1,  65536, 4096,  True,  "none"),
            (4,   4096, 4096,  False, "none"),
            (4,   4096, 4096,  True,  "none"),
            (8,   4096, 4096,  False, "none"),
            (8,   4096, 4096,  True,  "none"),
            (8,   8192, 4096,  False, "none"),
            (8,   8192, 4096,  True,  "none"),
            (16,  4096, 4096,  False, "none"),
            (16,  4096, 4096,  True,  "none"),
            (32,  2048, 4096,  False, "none"),
            (32,  2048, 4096,  True,  "none"),
            (8,   4096, 12288, False, "none"),
            (8,   4096, 12288, True,  "none"),
            (64,  1024, 4096,  False, "none"),
            (64,  1024, 4096,  True,  "none"),
            # --- Real workload shape ---
            (16,  6144, 2880,  False, "none"),
            (16,  6144, 2880,  True,  "none"),
            # --- Mild MoE-like imbalance ---
            (8,   4096, 4096,  False, "mild"),
            (8,   4096, 4096,  True,  "mild"),
            (16,  4096, 4096,  False, "mild"),
            (16,  4096, 4096,  True,  "mild"),
            (64,  1024, 4096,  False, "mild"),
            (64,  1024, 4096,  True,  "mild"),
            # --- Heavy hotspot imbalance ---
            (8,   4096, 4096,  False, "heavy"),
            (8,   4096, 4096,  True,  "heavy"),
            (16,  4096, 4096,  False, "heavy"),
            (16,  4096, 4096,  True,  "heavy"),
            # --- Zipf distribution ---
            (8,   4096, 4096,  False, "zipf"),
            (8,   4096, 4096,  True,  "zipf"),
            (16,  4096, 4096,  False, "zipf"),
            (16,  4096, 4096,  True,  "zipf"),
        ]
    else:
        configs = [
            (args.num_tensors, args.rows, args.hidden, args.use_scale, args.imbalance),
        ]

    dtype = DTYPE_MAP[args.dtype]
    header = (
        f"{'ntens':>5} {'rows':>6} {'hidden':>6} {'scale':>5} {'imbal':>5} "
        f"{'min_r':>6} {'max_r':>6} {'ratio':>5} {'total_MB':>9} "
        f"{'no_graph_ms':>11} {'no_graph_TB/s':>13} {'no_graph_%':>9} "
        f"{'graph_ms':>9} {'graph_TB/s':>11} {'graph_%':>7}"
    )
    print(header)
    print("-" * len(header))

    for num_tensors, rows, hidden, use_scale, imbalance in configs:
        output_gt, total_rows = make_grouped_output(
            num_tensors, rows, hidden, dtype, device, imbalance
        )
        bias_gt = make_grouped_bias(num_tensors, hidden, dtype, device)

        if use_scale:
            bias_scale = torch.randn(total_rows, dtype=torch.float32, device=device)
        else:
            bias_scale = torch.empty(0, dtype=torch.float32, device=device)

        # Compute per-expert row stats for display
        row_list = _generate_imbalanced_rows(num_tensors, rows, imbalance)
        min_r, max_r = min(row_list), max(row_list)
        ratio = max_r / max(min_r, 1)

        elem_size = torch.tensor([], dtype=dtype).element_size()
        total_mb = (total_rows * hidden * elem_size * 2
                    + num_tensors * hidden * elem_size
                    + (total_rows * 4 if use_scale else 0)) / 1e6

        # Without CUDA graph
        t_no_graph = bench_no_graph(output_gt, bias_gt, bias_scale, args.warmup, args.iters)
        bw_no_graph = compute_bandwidth_tb_s(
            total_rows, hidden, num_tensors, dtype, use_scale, t_no_graph
        )

        # With CUDA graph
        output_gt_g, _ = make_grouped_output(
            num_tensors, rows, hidden, dtype, device, imbalance
        )
        bias_gt_g = make_grouped_bias(num_tensors, hidden, dtype, device)
        if use_scale:
            bias_scale_g = torch.randn(total_rows, dtype=torch.float32, device=device)
        else:
            bias_scale_g = torch.empty(0, dtype=torch.float32, device=device)

        t_graph = bench_with_graph(output_gt_g, bias_gt_g, bias_scale_g, args.warmup, args.iters)
        bw_graph = compute_bandwidth_tb_s(
            total_rows, hidden, num_tensors, dtype, use_scale, t_graph
        )

        pct_no_graph = 100.0 * bw_no_graph / peak_bw
        pct_graph = 100.0 * bw_graph / peak_bw

        print(
            f"{num_tensors:>5} {rows:>6} {hidden:>6} "
            f"{'Y' if use_scale else 'N':>5} "
            f"{imbalance:>5} "
            f"{min_r:>6} {max_r:>6} {ratio:>5.1f} "
            f"{total_mb:>9.2f} "
            f"{t_no_graph:>11.4f} {bw_no_graph:>13.3f} {pct_no_graph:>8.1f}% "
            f"{t_graph:>9.4f} {bw_graph:>11.3f} {pct_graph:>6.1f}%"
        )

    print(f"\n{'='*90}")
    print(f"Peak HBM bandwidth: {peak_bw:.2f} TB/s")
    print(f"dtype: {args.dtype} ({torch.tensor([], dtype=dtype).element_size()} bytes)")
    print(
        "Note: Effective BW = (2*output_bytes + bias_bytes + scale_bytes) / elapsed_time\n"
        "      'graph' column uses CUDA graph replay to eliminate CPU overhead."
    )


if __name__ == "__main__":
    main()
