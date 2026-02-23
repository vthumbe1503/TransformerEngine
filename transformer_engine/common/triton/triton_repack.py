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
def _repack_tensor(
    in_ptr,
    out_ptr,
    split_tensor_ptr,
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

    group_offs = tl.arange(0, MAX_GROUPS)
    split_tensor = tl.load(
        split_tensor_ptr + group_offs,
        mask=group_offs < num_groups,
        other=0,
    ).to(tl.int64)

    sum_m128 = tl.sum(split_tensor, axis=0)
    last_prefix = sum_m128 * 4
    row_mask = rows < last_prefix
    col_mask = cols < k
    rows_safe = tl.where(row_mask, rows, 0)
    cols_safe = tl.where(col_mask, cols, 0)

    i = group_offs[:, None]
    j = group_offs[None, :]
    prefix_end_m128 = tl.sum(split_tensor[None, :] * (j <= i), axis=1)
    prefix_end = tl.where(group_offs < num_groups, prefix_end_m128 * 4, 2147483647)
    group = tl.sum(rows_safe[:, None] >= prefix_end[None, :], axis=1)

    prev = tl.maximum(group - 1, 0)
    start_m128 = tl.sum(
        tl.where(prev[:, None] == group_offs[None, :], prefix_end_m128[None, :], 0),
        axis=1,
    )
    start_m128 = tl.where(group > 0, start_m128, 0)
    start_m32 = start_m128 * 4
    m128_g = tl.sum(
        tl.where(group[:, None] == group_offs[None, :], split_tensor[None, :], 0),
        axis=1,
    )
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

    k = tl.full((), k, tl.int64)
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


def triton_repack_for_split_by_dim2_concat_along_dim0(
    tensor_to_split: torch.Tensor,
    split_tensor: torch.Tensor,
    k: int,
    total_rows: int,
    *,
    block_m: int = 16,
    block_n: int = 128,
    max_groups: int = 256,
) -> torch.Tensor:
    """Triton kernel equivalent to:
    torch.cat(torch.split(tensor_to_split, split_tensor.cpu(), dim=2), dim=0)
    Triton kernel is to avoid copy split_tensor to CPU so we dont break CUDA graph
    """

    out = torch.empty((total_rows, k), dtype=torch.uint8, device=tensor_to_split.device)
    tensor_to_split_view = tensor_to_split.view(torch.uint8)

    stride_b, stride_c, stride_d, stride_e, stride_f, stride_g = tensor_to_split_view.stride()
    stride_out_m, stride_out_n = out.stride()

    grid = (
        triton.cdiv(total_rows, block_m),
        triton.cdiv(k, block_n),
    )

    _repack_tensor[grid](
        tensor_to_split_view,
        out,
        split_tensor,
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
        int(split_tensor.numel()),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        MAX_GROUPS=max_groups,
    )
    return out
