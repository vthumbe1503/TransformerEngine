#!/usr/bin/env python3
"""Microbenchmark: measures peak throughput of a simple read-modify-write kernel.

    a[i] = a[i] + b[i] * c[i]

where a, b are bf16 and c is fp32.

Usage:
    python benchmarks/bench_rmw.py
"""

import argparse
import subprocess
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>

template <int kVec, int kBlockDim>
__global__ void rmw_kernel(__nv_bfloat16 *__restrict__ a,
                           const __nv_bfloat16 *__restrict__ b,
                           const float *__restrict__ c,
                           int64_t N) {
  using VecBf16 = int4;   // 16 bytes = 8 x bf16
  using VecF32  = float4; // 16 bytes = 4 x float

  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t elem = idx * kVec;
  if (elem >= N) return;

  VecBf16 a_vec = *reinterpret_cast<const VecBf16 *>(a + elem);
  __nv_bfloat16 *a_vals = reinterpret_cast<__nv_bfloat16 *>(&a_vec);

  VecBf16 b_vec = *reinterpret_cast<const VecBf16 *>(b + elem);
  __nv_bfloat16 *b_vals = reinterpret_cast<__nv_bfloat16 *>(&b_vec);

  VecF32 c_vec0 = *reinterpret_cast<const VecF32 *>(c + elem);
  VecF32 c_vec1 = *reinterpret_cast<const VecF32 *>(c + elem + 4);
  float *c_vals0 = reinterpret_cast<float *>(&c_vec0);
  float *c_vals1 = reinterpret_cast<float *>(&c_vec1);

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float ai = __bfloat162float(a_vals[i]);
    float bi = __bfloat162float(b_vals[i]);
    a_vals[i] = __float2bfloat16(fmaf(bi, c_vals0[i], ai));
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float ai = __bfloat162float(a_vals[4 + i]);
    float bi = __bfloat162float(b_vals[4 + i]);
    a_vals[4 + i] = __float2bfloat16(fmaf(bi, c_vals1[i], ai));
  }

  *reinterpret_cast<VecBf16 *>(a + elem) = a_vec;
}

void launch_rmw(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.scalar_type() == torch::kFloat32);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous());

  int64_t N = a.numel();
  constexpr int kVec = 8;
  constexpr int kBlockDim = 256;
  int64_t num_vecs = (N + kVec - 1) / kVec;
  int grid = (num_vecs + kBlockDim - 1) / kBlockDim;

  rmw_kernel<kVec, kBlockDim><<<grid, kBlockDim>>>(
      reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(b.data_ptr()),
      reinterpret_cast<const float *>(c.data_ptr()),
      N);
}
"""

CPP_SRC = r"""
void launch_rmw(torch::Tensor a, torch::Tensor b, torch::Tensor c);
"""


def get_gpu_peak_bandwidth_tb_s() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0]
        mem_total_mib, mem_clock_mhz = [float(x.strip()) for x in out.split(",")]
        if mem_total_mib > 60000:
            bus_width = 8192
        elif mem_total_mib > 30000:
            bus_width = 5120
        else:
            bus_width = 4096
        peak = 2.0 * mem_clock_mhz * 1e6 * bus_width / 8.0 / 1e12
        return peak
    except Exception:
        return 8.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    print("Compiling RMW kernel...")
    mod = load_inline(
        name="rmw_kernel",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["launch_rmw"],
        verbose=False,
    )

    peak_bw = get_gpu_peak_bandwidth_tb_s()
    print(f"Estimated peak HBM bandwidth: {peak_bw:.1f} TB/s\n")

    sizes = [
        (4096, 4096),
        (16384, 4096),
        (32768, 4096),
        (65536, 4096),
        (4096, 12288),
        (65536, 12288),
    ]

    print(f"{'rows':>8} {'cols':>8} {'total_MB':>10} {'time_ms':>10} {'TB/s':>8} {'%peak':>8}")
    print("-" * 60)

    for rows, cols in sizes:
        N = rows * cols
        a = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(N, device="cuda", dtype=torch.float32)

        # Total traffic: read a (bf16) + read b (bf16) + read c (fp32) + write a (bf16)
        total_bytes = N * (2 + 2 + 4 + 2)
        total_mb = total_bytes / 1e6

        for _ in range(args.warmup):
            mod.launch_rmw(a, b, c)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            mod.launch_rmw(a, b, c)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / args.iters
        bw_tb_s = total_bytes / (elapsed_ms * 1e-3) / 1e12
        pct = bw_tb_s / peak_bw * 100.0

        print(f"{rows:>8} {cols:>8} {total_mb:>10.2f} {elapsed_ms:>10.4f} {bw_tb_s:>8.3f} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
