# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Standalone throughput benchmark for cuDNN CuTe DSL grouped GEMM wgrad kernel.

Measures the isolated wgrad GEMM throughput for grouped MLP sizes matching
DeepSeek-V3 style MoE models:
  - FC1 wgrad: (out_features=7168, in_features=4096)
  - FC2 wgrad: (out_features=2048, in_features=7168)

Usage:
    NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python benchmarks/linear/benchmark_cudnn_wgrad.py

    # With custom token counts and experts:
    NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python benchmarks/linear/benchmark_cudnn_wgrad.py \
        --total-tokens 32768 65536 --num-experts 8

    # Compare 1CTA vs 2CTA:
    NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python benchmarks/linear/benchmark_cudnn_wgrad.py \
        --compare-1cta
"""

import argparse
import sys
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

import torch



def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _create_fp8_tensor(shape, dtype=torch.float8_e4m3fn):
    n = 1
    for s in shape:
        n *= s
    return torch.randint(-1, 2, (n,), dtype=torch.bfloat16, device="cuda").to(dtype).reshape(shape)


def _create_scale_tensor(shape, dtype=torch.float8_e8m0fnu):
    n = 1
    for s in shape:
        n *= s
    return torch.randint(1, 3, (n,), dtype=torch.float32, device="cuda").to(dtype).reshape(shape)


def _to_blocked(scale_2d):
    """Swizzle 2D scale tensor into the blocked layout expected by cuDNN."""
    rows, cols = scale_2d.shape
    row_blocks = _ceil_div(rows, 128)
    col_blocks = _ceil_div(cols, 4)
    padded_rows = row_blocks * 128
    padded_cols = col_blocks * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device)
        padded[:rows, :cols] = scale_2d
    else:
        padded = scale_2d
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _cat_byte(tensors, dim=0):
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        return torch.cat([t.view(torch.uint8) for t in tensors], dim=dim).view(first.dtype)
    return torch.cat(tensors, dim=dim)


def _assemble_scales(raw_scales, non_k_size):
    flat_parts = [_to_blocked(s) for s in raw_scales]
    all_flat = _cat_byte(flat_parts, dim=0)
    return all_flat.reshape(_round_up(non_k_size, 128), -1)


def build_wgrad_inputs(
    out_features: int,
    in_features: int,
    total_tokens: int,
    num_experts: int,
    sf_vec_size: int = MXFP8_BLOCK_SCALING_SIZE,
):
    """Build synthetic FP8 tensors for the wgrad kernel.

    Returns (a_tensor, b_tensor, sfa_tensor, sfb_tensor, offsets_tensor).
    """
    tokens_per_expert = total_tokens // num_experts
    remainder = total_tokens - tokens_per_expert * num_experts
    split_sizes = [tokens_per_expert] * num_experts
    for i in range(remainder):
        split_sizes[i] += 1

    ab_dtype = torch.float8_e4m3fn
    sf_dtype = torch.float8_e8m0fnu

    a_tensor = _create_fp8_tensor((out_features, total_tokens), ab_dtype)
    b_tensor = _create_fp8_tensor((total_tokens, in_features), ab_dtype).T.contiguous().T

    offsets_tensor = torch.cumsum(
        torch.tensor(split_sizes, device="cuda"), dim=0,
    ).to(torch.int32)

    raw_sfa = [
        _create_scale_tensor((out_features, _ceil_div(k, sf_vec_size)), sf_dtype)
        for k in split_sizes
    ]
    raw_sfb = [
        _create_scale_tensor((in_features, _ceil_div(k, sf_vec_size)), sf_dtype)
        for k in split_sizes
    ]
    sfa_tensor = _assemble_scales(raw_sfa, out_features)
    sfb_tensor = _assemble_scales(raw_sfb, in_features)

    return a_tensor, b_tensor, sfa_tensor, sfb_tensor, offsets_tensor


def benchmark_wgrad_kernel(
    kernel_fn,
    out_features: int,
    in_features: int,
    total_tokens: int,
    num_experts: int,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    """Benchmark a single wgrad configuration and return median time in ms."""
    a, b, sfa, sfb, offsets = build_wgrad_inputs(
        out_features, in_features, total_tokens, num_experts,
    )

    kwargs = dict(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        offsets_tensor=offsets,
        output_mode="dense",
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
        mma_tiler_mn=mma_tiler_mn,
        sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
    )
    if cluster_shape_mn is not None:
        kwargs["cluster_shape_mn"] = cluster_shape_mn

    # Warm-up: compile + warm caches
    for _ in range(warmup_iters):
        kernel_fn(**kwargs)
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        kernel_fn(**kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    return median_ms, min_ms, max_ms


def compute_tflops(out_features, in_features, total_tokens, time_ms):
    """Compute effective TFLOPS for grouped wgrad GEMMs.

    Each expert computes: wgrad[e] = DY_e^T @ X_e, where DY_e is
    (tokens_e, out_features) and X_e is (tokens_e, in_features).
    Total FLOPs = 2 * out_features * in_features * total_tokens.
    """
    flops = 2.0 * out_features * in_features * total_tokens
    return flops / (time_ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuDNN CuTe DSL grouped GEMM wgrad kernel",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536],
        help="Total token counts to benchmark",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts (groups)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--compare-1cta",
        action="store_true",
        help="Also benchmark 128x128 tiler mode (cluster=1x1)",
    )
    args = parser.parse_args()

    try:
        from cudnn import grouped_gemm_wgrad_wrapper_sm100
    except ImportError:
        print("ERROR: cudnn.grouped_gemm_wgrad_wrapper_sm100 not available.")
        print("Make sure cudnn-frontend is installed with SM100 support.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        print(f"ERROR: Requires SM100+, found SM{major * 10 + minor}.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"Num experts: {args.num_experts}")
    print(f"Warmup iters: {args.warmup_iters}, Bench iters: {args.bench_iters}")
    print()

    # FC1 wgrad: weight shape (7168, 4096) => out_features=7168, in_features=4096
    # FC2 wgrad: weight shape (2048, 7168) => out_features=2048, in_features=7168
    gemm_configs = [
        ("FC1 wgrad", 7168, 4096),
        ("FC2 wgrad", 2048, 7168),
    ]

    # (label, mma_tiler_mn, cluster_shape_mn)
    configs = [
        ("256x256 cl=2x1", (256, 256), (2, 1)),
        ("256x256 cl=2x2", (256, 256), (2, 2)),
    ]
    if args.compare_1cta:
        configs.append(("128x128 cl=1x1", (128, 128), (1, 1)))

    for label, out_feat, in_feat in gemm_configs:
        print(f"{'=' * 80}")
        print(f"  {label}: out_features={out_feat}, in_features={in_feat}")
        print(f"{'=' * 80}")
        header = (
            f"{'Tokens':>8}  {'Config':>14}"
            f"  {'Median(ms)':>10}  {'Min(ms)':>9}  {'Max(ms)':>9}  {'TFLOPS':>8}"
        )
        print(header)
        print("-" * len(header))

        for total_tokens in args.total_tokens:
            if total_tokens < args.num_experts:
                print(f"{total_tokens:>8}  SKIPPED (fewer tokens than experts)")
                continue

            for mode_label, mma_tiler, cluster in configs:
                median_ms, min_ms, max_ms = benchmark_wgrad_kernel(
                    kernel_fn=grouped_gemm_wgrad_wrapper_sm100,
                    out_features=out_feat,
                    in_features=in_feat,
                    total_tokens=total_tokens,
                    num_experts=args.num_experts,
                    mma_tiler_mn=mma_tiler,
                    cluster_shape_mn=cluster,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                tflops = compute_tflops(out_feat, in_feat, total_tokens, median_ms)
                print(
                    f"{total_tokens:>8}  {mode_label:>14}"
                    f"  {median_ms:>10.3f}  {min_ms:>9.3f}  {max_ms:>9.3f}  {tflops:>8.2f}"
                )
        print()


if __name__ == "__main__":
    main()
