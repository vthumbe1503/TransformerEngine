# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused grouped-dbias (+optional dscales) Triton kernel."""

import triton
import triton.language as tl


@triton.jit
def _grouped_dbias_kernel(
    dy_ptr,
    dbias_ptr,
    offsets_ptr,
    scales_ptr,
    bias_ptr,
    dscales_ptr,
    hidden,
    HAS_SCALES: tl.constexpr,
    N_ROW_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grouped dbias, optionally fused with dscales.

    For tokens i in group g(i), with s_i = scales[i] if HAS_SCALES else 1::

        dbias[g, j] += sum_{i in g} dy[i, j] * s_i

    When HAS_SCALES is True, additionally::

        dscales[i] += sum_j dy[i, j] * bias[g(i), j]

    Grid: (num_groups, N_ROW_SPLITS, cdiv(hidden, BLOCK_H)).

    Each CTA computes its group's actual size from device-side offsets,
    splits the row tiles evenly across N_ROW_SPLITS, and loops over only
    its share -- dynamic loop bound, no host-device sync, no wasted iters.

    - dbias: register-accumulated, one atomic-add per CTA at the end
      (N_ROW_SPLITS contributors per group).
    - dscales (if enabled): atomic-add per column-tile iteration
      (cdiv(hidden, BLOCK_H) contributors per element).

    When HAS_SCALES is False, ``scales_ptr``, ``bias_ptr`` and
    ``dscales_ptr`` are unused and may be passed as dummy pointers.
    """
    group_idx = tl.program_id(0)
    row_split = tl.program_id(1)
    col_block = tl.program_id(2)

    row_start = tl.load(offsets_ptr + group_idx)
    row_end = tl.load(offsets_ptr + group_idx + 1)

    group_rows = row_end - row_start
    total_tiles = (group_rows + BLOCK_M - 1) // BLOCK_M
    tiles_per_split = (total_tiles + N_ROW_SPLITS - 1) // N_ROW_SPLITS
    my_tile_start = row_split * tiles_per_split
    col_offs = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = col_offs < hidden

    if HAS_SCALES:
        bias_vals = tl.load(
            bias_ptr + group_idx * hidden + col_offs,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

    dbias_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    row_offs = tl.arange(0, BLOCK_M)
    for local_tile in range(tiles_per_split):
        tile_idx = my_tile_start + local_tile
        global_rows = row_start + tile_idx * BLOCK_M + row_offs
        row_mask = global_rows < row_end
        tile_mask = row_mask[:, None] & col_mask[None, :]
        dy_tile = tl.load(
            dy_ptr + global_rows[:, None] * hidden + col_offs[None, :],
            mask=tile_mask,
            other=0.0,
        ).to(tl.float32)

        if HAS_SCALES:
            scales_vals = tl.load(scales_ptr + global_rows, mask=row_mask, other=0.0)
            dbias_acc += tl.sum(dy_tile * scales_vals[:, None], axis=0)

            dscales_partial = tl.sum(dy_tile * bias_vals[None, :], axis=1)
            tl.atomic_add(
                dscales_ptr + global_rows,
                dscales_partial,
                mask=row_mask,
            )
        else:
            dbias_acc += tl.sum(dy_tile, axis=0)

    tl.atomic_add(
        dbias_ptr + group_idx * hidden + col_offs,
        dbias_acc,
        mask=col_mask,
    )


@triton.jit
def _grouped_dbias_rowwise_kernel(
    dy_ptr,
    dbias_ptr,
    offsets_ptr,
    scales_ptr,
    bias_ptr,
    dscales_ptr,
    hidden,
    num_groups,
    HAS_SCALES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Row-wise grouped dbias, optionally fused with dscales.

    Grid: (row_workers, cdiv(hidden, BLOCK_H)).

    Each program owns a row chunk and one column tile. This keeps the row
    scheduling independent of ``num_groups`` while preserving column
    parallelism.
    """
    row_worker = tl.program_id(0)
    col_block = tl.program_id(1)
    row_worker_stride = tl.num_programs(0)

    total_rows = tl.load(offsets_ptr + num_groups)
    chunk_start = (row_worker * BLOCK_M).to(tl.int64)

    row_offs = tl.arange(0, BLOCK_M)
    col_offs = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = col_offs < hidden

    while chunk_start < total_rows:
        chunk_end = tl.minimum(chunk_start + BLOCK_M, total_rows)

        group_idx = 0
        group_end = tl.load(offsets_ptr + 1)
        while (group_idx + 1 < num_groups) & (group_end <= chunk_start):
            group_idx += 1
            group_end = tl.load(offsets_ptr + group_idx + 1)

        seg_start = chunk_start
        while seg_start < chunk_end:
            while (group_idx + 1 < num_groups) & (group_end <= seg_start):
                group_idx += 1
                group_end = tl.load(offsets_ptr + group_idx + 1)

            seg_end = tl.minimum(group_end, chunk_end)
            if HAS_SCALES:
                dscales_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

            global_rows = seg_start + row_offs
            row_mask = global_rows < seg_end
            if HAS_SCALES:
                scales_vals = tl.load(scales_ptr + global_rows, mask=row_mask, other=0.0).to(
                    tl.float32
                )

            if HAS_SCALES:
                bias_vals = tl.load(
                    bias_ptr + group_idx * hidden + col_offs,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)

            tile_mask = row_mask[:, None] & col_mask[None, :]
            dy_tile = tl.load(
                dy_ptr + global_rows[:, None] * hidden + col_offs[None, :],
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)

            if HAS_SCALES:
                dbias_acc = tl.sum(dy_tile * scales_vals[:, None], axis=0)
                dscales_acc += tl.sum(dy_tile * bias_vals[None, :], axis=1)
            else:
                dbias_acc = tl.sum(dy_tile, axis=0)

            tl.atomic_add(
                dbias_ptr + group_idx * hidden + col_offs,
                dbias_acc,
                mask=col_mask,
            )

            if HAS_SCALES:
                tl.atomic_add(dscales_ptr + global_rows, dscales_acc, mask=row_mask)

            seg_start = seg_end

        chunk_start += (row_worker_stride * BLOCK_M).to(tl.int64)


