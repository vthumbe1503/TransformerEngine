#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for repacking fused MLP scale buffers."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _repack_swiglu_fc2_col_scale_kernel(
    in_ptr,
    out_ptr,
    prefix_m32_ptr,
    prefix_m128_ptr,
    split_m128_ptr,
    stride_b,
    stride_c,
    stride_d,
    stride_e,
    stride_f,
    stride_g,
    stride_out_m,
    stride_out_n,
    total_rows,
    k,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    last_prefix = tl.load(prefix_m32_ptr + (num_groups - 1))
    row_mask = rows < last_prefix
    col_mask = cols < k
    rows_safe = tl.where(row_mask, rows, 0)
    cols_safe = tl.where(col_mask, cols, 0)

    group_offs = tl.arange(0, MAX_GROUPS)
    prefix_end = tl.load(
        prefix_m32_ptr + group_offs,
        mask=group_offs < num_groups,
        other=2147483647,
    )
    group = tl.sum(rows_safe[:, None] >= prefix_end[None, :], axis=1)

    prev = tl.maximum(group - 1, 0)
    start_m32 = tl.load(prefix_m32_ptr + prev, mask=group > 0, other=0)
    start_m128 = tl.load(prefix_m128_ptr + prev, mask=group > 0, other=0)
    m128_g = tl.load(split_m128_ptr + group, mask=group < num_groups, other=0)
    m128_g_safe = tl.maximum(m128_g, 1)

    local_row = rows_safe - start_m32

    stride_b = tl.full((), stride_b, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_d = tl.full((), stride_d, tl.int64)
    stride_e = tl.full((), stride_e, tl.int64)
    stride_f = tl.full((), stride_f, tl.int64)
    stride_g = tl.full((), stride_g, tl.int64)
    stride_out_m = tl.full((), stride_out_m, tl.int64)
    stride_out_n = tl.full((), stride_out_n, tl.int64)

    linear = local_row[:, None].to(tl.int64) * k + cols_safe[None, :].to(tl.int64)
    k128 = k // 128
    g = linear % 4
    linear = linear // 4
    f = linear % 4
    linear = linear // 4
    e = linear % 32
    linear = linear // 32
    d_local = linear % m128_g_safe[:, None]
    linear = linear // m128_g_safe[:, None]
    c_local = linear % k128
    b = linear // k128
    d = d_local + start_m128[:, None]
    c = c_local

    in_ptrs = (
        in_ptr
        + b.to(tl.int64) * stride_b
        + c.to(tl.int64) * stride_c
        + d.to(tl.int64) * stride_d
        + e.to(tl.int64) * stride_e
        + f.to(tl.int64) * stride_f
        + g.to(tl.int64) * stride_g
    )
    mask = row_mask[:, None] & col_mask[None, :] & (m128_g[:, None] > 0)
    vals = tl.load(in_ptrs, mask=mask, other=0)

    out_ptrs = (
        out_ptr
        + rows.to(tl.int64)[:, None] * stride_out_m
        + cols.to(tl.int64)[None, :] * stride_out_n
    )
    tl.store(out_ptrs, vals, mask=mask)


def repack_swiglu_fc2_col_scale(
    fc2_in_col_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    k: int,
    total_rows: int,
    *,
    block_m: int = 16,
    block_n: int = 128,
    max_groups: int = 256,
) -> torch.Tensor:
    """Repack columnwise scales to match split-by-dim2 + view(-1, k) layout on GPU."""
    if not fc2_in_col_scale.is_cuda:
        raise ValueError("fc2_in_col_scale must be on CUDA for Triton repack.")
    if split_sizes.device.type != "cuda":
        raise ValueError("split_sizes must be on CUDA for Triton repack.")
    if int(split_sizes.numel()) > max_groups:
        raise ValueError(
            f"num_groups={int(split_sizes.numel())} exceeds max_groups={max_groups}"
        )
    if split_sizes.dtype != torch.int32:
        split_sizes_i32 = split_sizes.to(dtype=torch.int32)
    else:
        split_sizes_i32 = split_sizes

    split_m128 = split_sizes_i32 // 128
    split_m32 = split_m128 * 4
    prefix_m32 = torch.cumsum(split_m32, dim=0)
    prefix_m128 = torch.cumsum(split_m128, dim=0)

    out = torch.empty((total_rows, k), dtype=fc2_in_col_scale.dtype, device=fc2_in_col_scale.device)

    in_is_float8 = "float8" in str(fc2_in_col_scale.dtype)
    if in_is_float8:
        fc2_in_col_scale_view = fc2_in_col_scale.view(torch.uint8)
        out_view = out.view(torch.uint8)
    else:
        fc2_in_col_scale_view = fc2_in_col_scale
        out_view = out

    stride_b, stride_c, stride_d, stride_e, stride_f, stride_g = fc2_in_col_scale_view.stride()
    stride_out_m, stride_out_n = out.stride()

    grid = (
        triton.cdiv(total_rows, block_m),
        triton.cdiv(k, block_n),
    )

    _repack_swiglu_fc2_col_scale_kernel[grid](
        fc2_in_col_scale_view,
        out_view,
        prefix_m32,
        prefix_m128,
        split_m128,
        stride_b,
        stride_c,
        stride_d,
        stride_e,
        stride_f,
        stride_g,
        stride_out_m,
        stride_out_n,
        total_rows,
        k,
        int(split_sizes_i32.numel()),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        MAX_GROUPS=max_groups,
    )
    return out
