# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""GEMM preparation accepts column-padded compact MXFP8 scales.

Compact scales keep the contiguous dimension tile-aligned and drop padding on the
outer dimension, which is the layout left after an FSDP2 all-gather:

  rowwise:    (M, round_up(ceil(K/32), 4))
  columnwise: (ceil(M/32), round_up(K, 128))
"""

import pytest
import torch
from mxfp8_utils import swizzle_mxfp8_scale

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _mxfp8_scale_shape(rows, cols, columnwise, compact):
    """Scale shape with the contiguous dimension padded and, if compact, rows unpadded."""
    if columnwise:
        scale_rows = (rows + 31) // 32
        scale_cols = _round_up(cols, 128)
        if not compact:
            scale_rows = _round_up(scale_rows, 4)
        return scale_rows, scale_cols
    scale_cols = _round_up((cols + 31) // 32, 4)
    scale_rows = rows if compact else _round_up(rows, 128)
    return scale_rows, scale_cols


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize("shape", [(64, 32), (64, 96), (96, 160), (160, 64), (256, 128)])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("compact", [False, True])
def test_compact_scales_gemm(shape, layout, compact):
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    torch.manual_seed(1234)
    m, n = shape
    # general_gemm uses column-major operand conventions.
    b_shape = (96, n) if layout == "TN" else (96, m) if layout == "NN" else (m, 96)
    inputs = [torch.randn(s, device="cuda", dtype=torch.float32) for s in (shape, b_shape)]
    references = [quantizer(x) for x in inputs]
    operands = []
    snapshots = []
    for x, reference in zip(inputs, references):
        rows, cols = x.shape
        row_shape = _mxfp8_scale_shape(rows, cols, columnwise=False, compact=compact)
        col_shape = _mxfp8_scale_shape(rows, cols, columnwise=True, compact=compact)
        tensor = MXFP8Tensor(
            shape=x.shape,
            dtype=torch.bfloat16,
            device=x.device,
            rowwise_data=torch.empty_like(x, dtype=torch.uint8),
            columnwise_data=torch.empty_like(x, dtype=torch.uint8),
            rowwise_scale_inv=torch.empty(row_shape, device=x.device, dtype=torch.uint8),
            columnwise_scale_inv=torch.empty(col_shape, device=x.device, dtype=torch.uint8),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
        )
        tensor._rowwise_data.copy_(reference._rowwise_data)
        tensor._columnwise_data.copy_(reference._columnwise_data)
        tensor._rowwise_scale_inv.copy_(
            reference._rowwise_scale_inv[: row_shape[0], : row_shape[1]]
        )
        tensor._columnwise_scale_inv.copy_(
            reference._columnwise_scale_inv[: col_shape[0], : col_shape[1]]
        )
        operands.append(tensor)
        snapshots.append(
            [
                (s.data_ptr(), s.clone())
                for s in (tensor._rowwise_scale_inv, tensor._columnwise_scale_inv)
            ]
        )
    expected = general_gemm(*references, out_dtype=torch.bfloat16, layout=layout)[0]
    actual = general_gemm(*operands, out_dtype=torch.bfloat16, layout=layout)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for tensor, snapshot in zip(operands, snapshots):
        assert not tensor._with_gemm_swizzled_scales
        for scale, (pointer, before) in zip(
            (tensor._rowwise_scale_inv, tensor._columnwise_scale_inv), snapshot
        ):
            assert scale.data_ptr() == pointer
            torch.testing.assert_close(scale, before, rtol=0, atol=0)


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize(
    "shape",
    [
        # Column pad is non-trivial (ceil(K/32) or K is not already tile-aligned)
        # and the outer dimension is not tile-aligned, including the 1-column
        # rowwise case that is not 4-byte aligned without the column pad.
        (32, 32),
        (64, 32),
        (64, 96),
        (96, 160),
        (128, 128),
        (160, 96),
        (288, 224),
        (96, 4096),
        (64, 4224),
        (4096, 4128),
    ],
)
@pytest.mark.parametrize("rowwise", [False, True])
def test_compact_scale_swizzle_bytes(shape, rowwise):
    m, n = shape
    scale_rows, scale_cols = _mxfp8_scale_shape(m, n, columnwise=not rowwise, compact=True)
    logical_rows = m if rowwise else (m + 31) // 32
    logical_cols = (n + 31) // 32 if rowwise else n
    scales = torch.zeros((scale_rows, scale_cols), dtype=torch.uint8, device="cuda")
    scales[:logical_rows, :logical_cols] = torch.randint(
        0, 256, (logical_rows, logical_cols), dtype=torch.uint8, device="cuda"
    )
    padded_m, padded_n = _round_up(m, 128), _round_up(n, 128)
    padded_shape = _mxfp8_scale_shape(padded_m, padded_n, columnwise=not rowwise, compact=False)
    padded = torch.zeros(padded_shape, dtype=torch.uint8, device="cuda")
    padded[:scale_rows, :scale_cols].copy_(scales)
    expected = swizzle_mxfp8_scale(padded_m, padded_n, padded, columnwise=not rowwise)
    tensor = MXFP8Tensor(
        shape=shape,
        dtype=torch.bfloat16,
        device="cuda",
        rowwise_data=(torch.empty(shape, dtype=torch.uint8, device="cuda") if rowwise else None),
        columnwise_data=(None if rowwise else torch.empty(shape, dtype=torch.uint8, device="cuda")),
        rowwise_scale_inv=scales if rowwise else None,
        columnwise_scale_inv=None if rowwise else scales,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise),
        with_gemm_swizzled_scales=False,
    )
    tex.swizzle_scales_for_gemm_(tensor)
    actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tensor._with_gemm_swizzled_scales


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize("shape", [(160, 96), (288, 224)])
@pytest.mark.parametrize("rowwise", [False, True])
def test_multi_tensor_compact_scale_swizzle_bytes(shape, rowwise):
    tensors = []
    expected = []
    m, n = shape
    for _ in range(3):
        scale_rows, scale_cols = _mxfp8_scale_shape(
            m, n, columnwise=not rowwise, compact=True
        )
        logical_rows = m if rowwise else (m + 31) // 32
        logical_cols = (n + 31) // 32 if rowwise else n
        scales = torch.zeros((scale_rows, scale_cols), dtype=torch.uint8, device="cuda")
        scales[:logical_rows, :logical_cols] = torch.randint(
            0, 256, (logical_rows, logical_cols), dtype=torch.uint8, device="cuda"
        )
        padded_m, padded_n = _round_up(m, 128), _round_up(n, 128)
        padded_shape = _mxfp8_scale_shape(
            padded_m, padded_n, columnwise=not rowwise, compact=False
        )
        padded = torch.zeros(padded_shape, dtype=torch.uint8, device="cuda")
        padded[:scale_rows, :scale_cols].copy_(scales)
        expected.append(
            swizzle_mxfp8_scale(padded_m, padded_n, padded, columnwise=not rowwise)
        )
        tensors.append(
            MXFP8Tensor(
                shape=shape,
                dtype=torch.bfloat16,
                device="cuda",
                rowwise_data=(
                    torch.empty(shape, dtype=torch.uint8, device="cuda") if rowwise else None
                ),
                columnwise_data=(
                    None if rowwise else torch.empty(shape, dtype=torch.uint8, device="cuda")
                ),
                rowwise_scale_inv=scales if rowwise else None,
                columnwise_scale_inv=None if rowwise else scales,
                fp8_dtype=tex.DType.kFloat8E4M3,
                quantizer=MXFP8Quantizer(
                    tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise
                ),
                with_gemm_swizzled_scales=False,
            )
        )

    tex.multi_tensor_swizzle_scales_for_gemm_(
        tensors, rowwise_usage=rowwise, columnwise_usage=not rowwise
    )
    for tensor, reference in zip(tensors, expected):
        actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
