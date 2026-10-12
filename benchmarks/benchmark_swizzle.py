#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CUDA-graph benchmark for MXFP8/NVFP4 scale swizzling.

Run this script from two builds and compare the JSON files:

  python benchmarks/benchmark_swizzle.py --preset full --output baseline.json
  python benchmarks/benchmark_swizzle.py --preset full --output candidate.json
  python benchmarks/benchmark_swizzle.py --compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


@dataclass(frozen=True)
class Case:
    scaling: str
    direction: str
    m: int
    k: int
    count: int = 1
    compact: bool = False

    @property
    def name(self) -> str:
        layout = "compact" if self.compact else "padded"
        return f"{self.scaling}_{self.direction}_{layout}_{self.m}x{self.k}_n{self.count}"


def cases_for_preset(preset: str, include_compact: bool) -> list[Case]:
    cases = [
        Case("mxfp8", "rowwise", 128, 128),
        Case("mxfp8", "rowwise", 4096, 4096),
        Case("mxfp8", "rowwise", 160, 8064),  # below the 32-KiB boundary
        Case("mxfp8", "rowwise", 160, 8192),  # above it after bank-conflict padding
        Case("mxfp8", "columnwise", 128, 128),
        Case("mxfp8", "columnwise", 4224, 2048),
        Case("nvfp4", "rowwise", 128, 128),
        Case("nvfp4", "columnwise", 128, 128),
    ]
    if include_compact:
        cases += [
            Case("mxfp8", "rowwise", 160, 96, compact=True),
            Case("mxfp8", "columnwise", 160, 96, compact=True),
        ]
    if preset == "full":
        cases += [
            Case("mxfp8", direction, m, k, count)
            for direction in ("rowwise", "columnwise")
            for m, k in (
                (32, 32),
                (64, 96),
                (256, 256),
                (4096, 128),
                (128, 8192),
                (16384, 128),
            )
            for count in (1, 4)
        ]
        cases += [
            Case("mxfp8", "rowwise", 128, 128, 80),
            Case("mxfp8", "columnwise", 128, 128, 80),
            Case("mxfp8", "columnwise", 4224, 2048, 20),
            Case("nvfp4", "rowwise", 4096, 4096),
            Case("nvfp4", "columnwise", 4096, 4096),
            Case("nvfp4", "rowwise", 128, 128, 4),
            Case("nvfp4", "columnwise", 128, 128, 4),
        ]
        if include_compact:
            cases += [
                Case("mxfp8", "rowwise", 288, 224, 4, compact=True),
                Case("mxfp8", "columnwise", 288, 224, 4, compact=True),
            ]
    return cases


def make_quantizer(case: Case):
    rowwise = case.direction == "rowwise"
    if case.scaling == "mxfp8":
        quantizer = te.MXFP8Quantizer(tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=rowwise, columnwise=not rowwise)
    else:
        quantizer = NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=rowwise,
            columnwise=not rowwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=True,
            with_post_rht_amax=True,
            with_random_sign_mask=True,
        )
    quantizer.optimize_for_gemm = False
    return quantizer


def make_tensors(case: Case):
    quantizer = make_quantizer(case)
    tensors = []
    for _ in range(case.count):
        quantized = quantizer(
            torch.randn(case.m, case.k, dtype=torch.bfloat16, device="cuda")
        )
        if not case.compact:
            tensors.append(quantized)
            continue
        if case.scaling != "mxfp8":
            raise ValueError("Compact benchmark construction is currently MXFP8-only")
        rowwise = case.direction == "rowwise"
        if rowwise:
            scales = quantized._rowwise_scale_inv[
                : case.m, : ((case.k + 31) // 32 + 3) // 4 * 4
            ].contiguous()
        else:
            scales = quantized._columnwise_scale_inv[
                : (case.m + 31) // 32, : (case.k + 127) // 128 * 128
            ].contiguous()
        tensors.append(
            MXFP8Tensor(
                shape=(case.m, case.k),
                dtype=torch.bfloat16,
                device="cuda",
                rowwise_data=quantized._rowwise_data if rowwise else None,
                columnwise_data=None if rowwise else quantized._columnwise_data,
                rowwise_scale_inv=scales if rowwise else None,
                columnwise_scale_inv=None if rowwise else scales,
                fp8_dtype=tex.DType.kFloat8E4M3,
                quantizer=quantizer,
                with_gemm_swizzled_scales=False,
            )
        )
    return tensors


def launch(tensors: list, case: Case) -> None:
    rowwise = case.direction == "rowwise"
    if len(tensors) == 1:
        tex.swizzle_scales_for_gemm_(tensors[0])
    else:
        tex.multi_tensor_swizzle_scales_for_gemm_(
            tensors, rowwise_usage=rowwise, columnwise_usage=not rowwise
        )


def benchmark_case(
    case: Case, warmup: int, iterations: int, calls_per_graph: int
) -> dict:
    # Initialize CUDA contexts and allocator paths on a side stream with objects
    # that are not reused for graph capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch(make_tensors(case), case)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph_tensors = [make_tensors(case) for _ in range(calls_per_graph)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for tensors in graph_tensors:
            launch(tensors, case)

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)
    total_calls = iterations * calls_per_graph
    result = asdict(case)
    result.update(
        {
            "name": case.name,
            "iterations": iterations,
            "calls_per_graph": calls_per_graph,
            "timing": "cuda_graph_amortized",
            "total_ms": total_ms,
            "us_per_call": total_ms * 1000.0 / total_calls,
            "us_per_tensor": total_ms * 1000.0 / (total_calls * case.count),
        }
    )
    return result


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metadata() -> dict:
    device = torch.cuda.current_device()
    return {
        "git_sha": git_sha(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(
    baseline_path: Path, candidate_path: Path, output_path: Path, csv_path: Path | None
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_by_name = {row["name"]: row for row in baseline["results"]}
    rows = []
    for current in candidate["results"]:
        previous = baseline_by_name.get(current["name"])
        if previous is None:
            continue
        old_us = previous["us_per_call"]
        new_us = current["us_per_call"]
        rows.append(
            {
                "name": current["name"],
                "baseline_us": old_us,
                "candidate_us": new_us,
                "speedup": old_us / new_us,
                "delta_percent": 100.0 * (new_us - old_us) / old_us,
            }
        )
    for row in rows:
        print(
            f"{row['name']:45s} {row['baseline_us']:9.3f} -> "
            f"{row['candidate_us']:9.3f} us  {row['speedup']:6.3f}x "
            f"({row['delta_percent']:+7.2f}%)"
        )
    payload = {
        "baseline_metadata": baseline.get("metadata", {}),
        "candidate_metadata": candidate.get("metadata", {}),
        "results": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path or output_path.with_suffix(".csv"), rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--calls-per-graph",
        type=int,
        default=16,
        help="Swizzle calls captured per graph to amortize graph replay overhead",
    )
    parser.add_argument(
        "--include-compact",
        action="store_true",
        help="Include compact outer-dimension cases (unsupported by the pre-PR baseline)",
    )
    parser.add_argument("--output", type=Path, default=Path("swizzle_benchmark.json"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), type=Path)
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1], args.output, args.csv)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    results = []
    for case in cases_for_preset(args.preset, args.include_compact):
        row = benchmark_case(case, args.warmup, args.iterations, args.calls_per_graph)
        results.append(row)
        print(f"{case.name:45s} {row['us_per_call']:9.3f} us/call")
    payload = {"metadata": metadata(), "preset": args.preset, "results": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv or args.output.with_suffix(".csv"), results)


if __name__ == "__main__":
    main()

