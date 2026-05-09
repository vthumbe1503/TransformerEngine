# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import functools
import inspect
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_gemm, general_grouped_gemm_for_grouped_tensor
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer, NVFP4Tensor, Quantizer
from ...utils import get_cached_ones_tensor, get_device_compute_capability, mark_grouped_tensor
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    fuse_grouped_mlp_ops,
    is_quantized_tensor,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)


def _pack_nvfp4_amax_list(tensors: list) -> None:
    """Ensure discrete NVFP4 weight list uses contiguous per-group amax buffers.

    The discrete-input grouped GEMM kernels expect a single contiguous device
    buffer for amax pointers across groups. This rebinds each tensor's
    ``_amax_rowwise`` / ``_amax_columnwise`` to a 1-element view into a packed
    buffer so that the resulting pointer array is contiguous in device memory.
    """
    if not tensors:
        return
    row_amaxes = [getattr(tensor, "_amax_rowwise", None) for tensor in tensors]
    if all(amax is not None for amax in row_amaxes):
        packed_row_amax = torch.cat([amax.view(-1) for amax in row_amaxes], dim=0).contiguous()
        for idx, tensor in enumerate(tensors):
            tensor._amax_rowwise = packed_row_amax[idx : idx + 1]
    col_amaxes = [getattr(tensor, "_amax_columnwise", None) for tensor in tensors]
    if all(amax is not None for amax in col_amaxes):
        packed_col_amax = torch.cat([amax.view(-1) for amax in col_amaxes], dim=0).contiguous()
        for idx, tensor in enumerate(tensors):
            tensor._amax_columnwise = packed_col_amax[idx : idx + 1]


def _debug_grouped_mlp_enabled() -> bool:
    if os.environ.get("NVTE_FUSED_GROUPED_MLP_DEBUG", "0") != "1":
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except RuntimeError:
        pass
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    debug_rank = int(os.environ.get("NVTE_FUSED_GROUPED_MLP_DEBUG_RANK", "0"))
    return debug_rank < 0 or rank == debug_rank


def _debug_grouped_mlp_tensor(name: str, tensor: torch.Tensor, *, prefix: str) -> None:
    if not _debug_grouped_mlp_enabled():
        return
    token_count = int(
        os.environ.get(
            "NVTE_FUSED_GROUPED_MLP_DEBUG_TOKENS",
            os.environ.get("NVTE_FUSED_DENSE_MLP_DEBUG_TOKENS", "4"),
        )
    )
    data = tensor.detach()
    if data.dim() > 1 and token_count > 0:
        data = data[:token_count]
    if os.environ.get("NVTE_FUSED_GROUPED_MLP_DEBUG_FULL", "0") == "1":
        data = tensor.detach()
    values = data.float()
    finite = torch.isfinite(values)
    sample = values.reshape(-1)[:8].cpu().tolist()
    print(
        f"{prefix} {name}: shape={tuple(tensor.shape)} sample_shape={tuple(data.shape)} "
        f"dtype={tensor.dtype} mean={values.mean().item():.8e} "
        f"std={values.std(unbiased=False).item():.8e} "
        f"min={values.min().item():.8e} max={values.max().item():.8e} "
        f"amax={values.abs().max().item():.8e} sum={values.sum().item():.8e} "
        f"finite={finite.sum().item()}/{finite.numel()} sample={sample}",
        flush=True,
    )


def _enable_nvfp4_rht_for_group_quantize(quantizer: Quantizer) -> None:
    """Use the graph-safe NVFP4 grouped quantization path.

    The current NVFP4 grouped quantize C++ implementation only supports the
    RHT path with post-RHT amax. Fused grouped MLP always uses graph-safe
    grouped quantize for activations, so enable the required quantizer flags
    locally even if the broader recipe disabled RHT.
    """
    if isinstance(quantizer, NVFP4Quantizer):
        quantizer.with_rht = True
        quantizer.with_post_rht_amax = True


def _wrap_single_nvfp4_as_grouped(
    tensor: torch.Tensor,
    quantized: Any,
    quantizer: NVFP4Quantizer,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Wrap a single NVFP4Tensor in GroupedTensor storage."""
    with_gemm_swizzled_scales = getattr(quantized, "_with_gemm_swizzled_scales", False)
    if getattr(quantizer, "optimize_for_gemm", False):
        tex.swizzle_scales_for_gemm_(quantized)
        with_gemm_swizzled_scales = True
    rowwise_data = getattr(quantized, "_rowwise_data", None)
    rowwise_scale = getattr(quantized, "_rowwise_scale_inv", None)
    columnwise_data = getattr(quantized, "_columnwise_data", None)
    columnwise_scale = getattr(quantized, "_columnwise_scale_inv", None)
    amax = getattr(quantized, "_amax_rowwise", None)
    columnwise_amax = getattr(quantized, "_amax_columnwise", None)

    if split_sizes is None:
        split_sizes = torch.full(
            (1,),
            tensor.shape[0],
            dtype=torch.int64,
            device=tensor.device,
        )
    else:
        split_sizes = split_sizes.to(dtype=torch.int64, device=tensor.device)

    m_dim = tensor.shape[0]
    if rowwise_data is not None:
        k_dim = rowwise_data.shape[-1] * 2
    elif columnwise_data is not None:
        k_dim = columnwise_data.shape[0]
    else:
        k_dim = tensor.shape[-1]

    if tensor_offsets is None:
        tensor_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=tensor.device),
                torch.cumsum(split_sizes * k_dim, dim=0),
            ],
        )

    return GroupedTensor(
        shape=(m_dim, k_dim),
        dtype=tensor.dtype,
        quantizer=quantizer,
        num_tensors=1,
        data=rowwise_data.reshape(-1) if rowwise_data is not None else None,
        columnwise_data=columnwise_data.reshape(-1) if columnwise_data is not None else None,
        scale_inv=rowwise_scale.reshape(-1) if rowwise_scale is not None else None,
        columnwise_scale_inv=columnwise_scale.reshape(-1)
        if columnwise_scale is not None
        else None,
        amax=amax,
        columnwise_amax=columnwise_amax,
        first_dims=split_sizes,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )


def _group_quantize_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize into grouped storage, using regular quantize for one-group NVFP4."""
    if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer):
        return tex.group_quantize(tensor, quantizer, num_groups, split_sizes)

    quantized = tex.quantize(tensor, quantizer)
    return _wrap_single_nvfp4_as_grouped(
        tensor,
        quantized,
        quantizer,
        split_sizes,
        tensor_offsets=tensor_offsets,
    )


def _group_quantize_with_amax_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    rowwise_amax: torch.Tensor,
    columnwise_amax: torch.Tensor,
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize with precomputed amaxes, using row_col_rht for one-group NVFP4."""
    if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer):
        return tex.group_quantize_with_amax(
            tensor,
            quantizer,
            num_groups,
            split_sizes,
            rowwise_amax,
            columnwise_amax,
        )

    quantized = tex.quantize_with_amax(
        tensor,
        quantizer,
        rowwise_amax.view(-1)[:1],
        columnwise_amax.view(-1)[:1],
    )
    return _wrap_single_nvfp4_as_grouped(
        tensor,
        quantized,
        quantizer,
        split_sizes,
        tensor_offsets=tensor_offsets,
    )


def _nvfp4_logical_data_view(data: torch.Tensor) -> torch.Tensor:
    """View packed NVFP4 data with its logical K dimension for scale swizzling."""
    return data.as_strided(
        (data.shape[0], data.shape[1] * 2),
        (data.stride(0), 0),
    )


def _nvfp4_rowwise_amax(tensors: Any) -> torch.Tensor:
    """Get one rowwise NVFP4 amax value per group."""
    if hasattr(tensors, "amax"):
        if tensors.amax is None:
            raise RuntimeError("NVFP4 GroupedTensor is missing rowwise amax.")
        return tensors.amax.view(-1)

    row_amaxes = [getattr(tensor, "_amax_rowwise", None) for tensor in tensors]
    if any(amax is None for amax in row_amaxes):
        raise RuntimeError("NVFP4 tensor list is missing rowwise amax.")
    return torch.cat([amax.view(-1) for amax in row_amaxes], dim=0)


def _pack_grouped_linear_bias_for_cudnn(linear_op: GroupedLinear) -> Optional[torch.Tensor]:
    """Bias layout expected by cuDNN grouped GEMM: shape (n, num_groups), stride (1, n)."""
    if not linear_op.has_bias:
        return None
    num_groups = linear_op.num_groups
    grouped_bias = getattr(linear_op, "bias", None)
    if grouped_bias is not None:
        packed = grouped_bias.rowwise_data.view(num_groups, -1)
        return packed.transpose(0, 1)
    rows = [getattr(linear_op, f"bias{group_idx}") for group_idx in range(num_groups)]
    # stack to [num_groups, n] but cuDNN expects [n, num_groups] with stride [1, n].
    return torch.stack(rows, dim=0).transpose(0, 1)


class ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8(FusedOperation):
    """Fused op for MXFP8 GroupedLinear + scaled GLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_glu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation, and post-multiplication."""
        from cudnn import grouped_gemm_glu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_glu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_glu_hadamard_kernel(cls) -> Optional[Callable]:
        """Fused grouped GEMM GLU kernel that also emits NVFP4 RHT amaxes."""
        try:
            from cudnn import (
                grouped_gemm_glu_hadamard_wrapper_sm100,
            )  # pylint: disable=no-name-in-module,import-outside-toplevel
        except ImportError:
            return None

        return grouped_gemm_glu_hadamard_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_quant_kernel(cls) -> Callable:
        """Grouped GEMM quant kernel for block-scaled inputs."""
        from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_quant_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        try:
            cls.grouped_gemm_glu_kernel()
            cls.grouped_gemm_quant_kernel()
        except ImportError:
            return False
        return True

    @classmethod
    @functools.lru_cache(maxsize=1)
    def is_fc1_bias_supported(cls) -> bool:
        """Whether cudnn-frontend exposes ``bias_tensor`` on the grouped GEMM GLU SM100 wrapper (FC1)."""
        if not cls.is_supported():
            return False
        try:
            from cudnn import (
                grouped_gemm_glu_wrapper_sm100,
            )  # pylint: disable=import-outside-toplevel
        except ImportError:
            return False
        try:
            params = inspect.signature(grouped_gemm_glu_wrapper_sm100).parameters
        except (TypeError, ValueError):
            return False
        return "bias_tensor" in params

    @classmethod
    @functools.lru_cache(maxsize=1)
    def is_fc2_bias_supported(cls) -> bool:
        """Whether cudnn-frontend exposes ``bias_tensor`` on the grouped GEMM Quant SM100 wrapper (FC2)."""
        if not cls.is_supported():
            return False
        try:
            from cudnn import (
                grouped_gemm_quant_wrapper_sm100,
            )  # pylint: disable=import-outside-toplevel
        except ImportError:
            return False
        try:
            params = inspect.signature(grouped_gemm_quant_wrapper_sm100).parameters
        except (TypeError, ValueError):
            return False
        return "bias_tensor" in params

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: ScaledSwiGLU | ScaledClampedQGeGLU,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((fc1, swiglu, fc2))
        if not self.is_supported():
            self.grouped_gemm_glu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, swiglu, fc2)
        # The cuDNN geglu implementation corresponds to ScaledClampedQGeGLU.
        # The act_func string should be fixed on the cuDNN FE side.
        self._cudnn_act_func: str = "geglu" if isinstance(swiglu, ScaledClampedQGeGLU) else "swiglu"

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        # Get basic operations
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, swiglu_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        input_ = input_.reshape(-1, fc1_weight_shape[1])
        in_shape = list(input_.size())
        assert in_shape[0] % 128 == 0, "Unsupported input shape for fused grouped MLP."

        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_weight_param.dtype

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = requires_grad and (
            fc1_weight_param.requires_grad or fc2_weight_param.requires_grad
        )

        # Quantizers
        fc1_input_quantizer = fc1_op.get_quantizer("forward", 0)
        fc1_weight_quantizer = fc1_op.get_quantizer("forward", 1)
        fc1_grad_output_quantizer = fc1_op.get_quantizer("backward", 0)
        fc2_input_quantizer = fc2_op.get_quantizer("forward", 0)
        fc2_weight_quantizer = fc2_op.get_quantizer("forward", 1)
        fc2_grad_output_quantizer = fc2_op.get_quantizer("backward", 0)

        # Extract split sizes from extra input
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            fc1_split_sizes.size() != fc2_split_sizes.size()
            or fc1_split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} got different split points for FC1 and FC2."
            )
        split_sizes = fc1_split_sizes
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        base_offsets = tex.splits_to_offsets(split_sizes, 1)
        split_points = base_offsets[1:].to(dtype=torch.int)
        fc1_x_tensor_offsets = base_offsets * fc1_weight_shape[1]
        fc2_x_tensor_offsets = base_offsets * fc2_weight_shape[1]

        # Extract post-scales from extra input
        scales = basic_op_extra_inputs[1][0]

        # Prepare FC1 grouped weight tensor for fused kernels.
        #  - single_grouped_weight=True: op.weight is already a GroupedTensor
        #  - single_grouped_weight=False: cute DSL kernel works with discrete weight tensors
        #   as long as host pointers for addresses are packed as contiguous device tensor.
        if fc1_op.single_grouped_weight:
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc1_op.weight.quantizer is not None:
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc1_op.weight.quantizer = fc1_weight_quantizer
                grouped_fc1_weight = fc1_op.weight
            else:
                if fc1_op.weight.rowwise_data is None:
                    raise RuntimeError("FC1 grouped weight has no rowwise_data to quantize.")
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc1_weight = _group_quantize_for_grouped_mlp(
                    fc1_op.weight.rowwise_data.view(fc1_op.weight.logical_shape),
                    fc1_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc1_weights = []
            for idx, weight in enumerate(fc1_weights):
                quantizer = fc1_op.get_quantizer("forward", 2 * idx + 1)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc1_weights.append(quantizer(weight))
                else:
                    quantized_fc1_weights.append(weight)
            grouped_fc1_weight = quantized_fc1_weights
            # NVFP4 discrete-input grouped GEMM requires per-group amax pointers
            # to be contiguous in device memory.
            if isinstance(fc1_input_quantizer, NVFP4Quantizer):
                _pack_nvfp4_amax_list(grouped_fc1_weight)

        # Prepare FC2 grouped weight tensor for fused kernels.
        if fc2_op.single_grouped_weight:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc2_op.weight.quantizer is not None:
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc2_op.weight.quantizer = fc2_weight_quantizer
                grouped_fc2_weight = fc2_op.weight
            else:
                if fc2_op.weight.rowwise_data is None:
                    raise RuntimeError("FC2 grouped weight has no rowwise_data to quantize.")
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc2_weight = _group_quantize_for_grouped_mlp(
                    fc2_op.weight.rowwise_data.view(fc2_op.weight.logical_shape),
                    fc2_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc2_weights = []
            for idx, weight in enumerate(fc2_weights):
                quantizer = fc2_op.get_quantizer("forward", 2 * idx + 1)
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc2_weights.append(quantizer(weight))
                else:
                    quantized_fc2_weights.append(weight)
            grouped_fc2_weight = quantized_fc2_weights
            # NVFP4 discrete-input grouped GEMM requires per-group amax pointers
            # to be contiguous in device memory.
            if isinstance(fc2_input_quantizer, NVFP4Quantizer):
                _pack_nvfp4_amax_list(grouped_fc2_weight)

        # Some wrapper-copy paths may drop grouped storage metadata; enforce defaults.
        if getattr(grouped_fc1_weight, "_with_gemm_swizzled_scales", None) is None and isinstance(
            grouped_fc1_weight, GroupedTensor
        ):
            grouped_fc1_weight._with_gemm_swizzled_scales = False
        if getattr(grouped_fc2_weight, "_with_gemm_swizzled_scales", None) is None and isinstance(
            grouped_fc2_weight, GroupedTensor
        ):
            grouped_fc2_weight._with_gemm_swizzled_scales = False

        # Group-quantize input tensor and convert dtypes if needed
        fc1_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        fc1_input_quantizer.optimize_for_gemm = True
        _enable_nvfp4_rht_for_group_quantize(fc1_input_quantizer)
        if isinstance(input_, GroupedTensor) and isinstance(
            getattr(input_, "quantizer", None), MXFP8Quantizer
        ):
            grouped_fc1_x = input_
        else:
            fc1_x = maybe_dequantize(input_, dtype)
            grouped_fc1_x = _group_quantize_for_grouped_mlp(
                fc1_x,
                fc1_input_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc1_x_tensor_offsets,
            )

        # NVFP4 vs MXFP8 data layout constants
        use_nvfp4 = isinstance(fc1_input_quantizer, NVFP4Quantizer) or (
            type(fc1_weight_param).__name__ == "NVFP4Tensor"
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        # NVFP4 byte-packs the K dimension (two FP4 values per byte).
        data_in_k = in_shape[1] // 2 if use_nvfp4 else in_shape[1]
        fc1_weight_k = fc1_weight_shape[1] // 2 if use_nvfp4 else fc1_weight_shape[1]
        # Number of FP4/FP8 values represented by one block scale along K.
        # For MXFP8: 4 * 32 = 128 (matches the 128-block tiling).
        # For NVFP4: 2 * 16 = 32 logical values = 16 byte-packed columns.
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        # For NVFP4, rowwise_data is byte-packed along K (K/2 storage).
        fc1_x_data = grouped_fc1_x.rowwise_data.view(dtype=data_dtype)
        fc1_x_data = fc1_x_data.view(in_shape[0], data_in_k)
        fc1_x_data = fc1_x_data.unsqueeze(0).permute(1, 2, 0)
        fc1_x_scales = grouped_fc1_x.scale_inv
        fc1_x_scales = fc1_x_scales.view(dtype=scale_view_dtype)
        with_gemm_swizzled_scales = getattr(
            grouped_fc1_x,
            "_with_gemm_swizzled_scales",
            getattr(grouped_fc1_x, "with_gemm_swizzled_scales", False),
        )
        if use_nvfp4 and with_gemm_swizzled_scales:
            # RHT kernel with optimize_for_gemm=True writes scales directly in
            # SwizzledSFALayout (cuDNN-compatible format). Only kernel-format permute needed.
            fc1_x_scales = fc1_x_scales.view(
                1,
                in_shape[0] // 128,
                data_in_k // k_sf_divisor,
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)
        elif use_nvfp4 and not with_gemm_swizzled_scales:
            # Unswizzled TE format: convert unswizzled to swizzled, then to kernel format.
            fc1_x_scales = fc1_x_scales.view(
                1,
                in_shape[0] // 128,
                4,
                32,
                data_in_k // k_sf_divisor,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 2, 1, 5, 4, 0)
        else:
            fc1_x_scales = fc1_x_scales.view(
                1,
                (in_shape[0] + 127) // 128,
                (in_shape[1] + k_sf_divisor - 1) // k_sf_divisor,
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)

        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, dtype, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        fc1_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc1_op)
        fc2_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc2_op)

        fc1_d_dtype = torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn
        fc1_prob_tensor = (
            scales.detach().to(dtype=torch.float32 if use_nvfp4 else dtype).reshape(-1, 1, 1)
        )
        fc1_norm_const_tensor = None if use_nvfp4 else norm_const_tensor
        if use_nvfp4:
            # Baseline cuBLAS approach: alpha = amax_A * amax_B / (fp4_max^2 * fp8_max^2).
            # This mirrors nvte_nvfp4_compute_per_tensor_scale for FP4's two-level scaling.
            _amax_x = _nvfp4_rowwise_amax(grouped_fc1_x)
            _amax_w = _nvfp4_rowwise_amax(grouped_fc1_weight)
            _nvfp4_fp4_max = 6.0
            _nvfp4_fp8_max = 448.0
            fc1_alpha_tensor = (
                _amax_x * _amax_w / (_nvfp4_fp4_max**2 * _nvfp4_fp8_max**2)
            ).to(torch.float32)
        else:
            fc1_alpha_tensor = alpha_tensor
        enable_fc1_glu_hadamard = (
            int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP_FC1_GLU_RHT_AMAX", "0")) > 0
        )
        use_tmem_post_rht_amax = (
            int(
                os.environ.get(
                    "NVTE_CUTEDSL_FUSED_GROUPED_MLP_FC1_GLU_RHT_AMAX_TMEM",
                    "0",
                )
            )
            > 0
        )
        fc1_glu_hadamard_kernel = (
            self.grouped_gemm_glu_hadamard_kernel() if enable_fc1_glu_hadamard else None
        )
        has_precomputed_amax_quantize = (
            hasattr(tex, "quantize_with_amax")
            if num_groups == 1
            else hasattr(tex, "group_quantize_with_amax")
        )
        use_fc1_glu_hadamard = (
            use_nvfp4
            and fc1_glu_hadamard_kernel is not None
            and has_precomputed_amax_quantize
        )
        fc1_glu_kwargs = {
            "a_tensor": fc1_x_data,
            "sfa_tensor": fc1_x_scales,
            "padded_offsets": split_points,
            "alpha_tensor": fc1_alpha_tensor,
            "bias_tensor": fc1_bias_packed,
            "prob_tensor": fc1_prob_tensor,
            "acc_dtype": torch.float32,
            "c_dtype": torch.bfloat16,
            "d_dtype": fc1_d_dtype,
            "cd_major": "n",
            "sf_vec_size": sf_vec_size,
            "current_stream": current_stream,
            "act_func": self._cudnn_act_func,
            "use_dynamic_sched": True,
        }
        if use_fc1_glu_hadamard:
            fc1_glu_kwargs["use_tmem_post_rht_amax"] = use_tmem_post_rht_amax
        else:
            fc1_glu_kwargs["norm_const_tensor"] = fc1_norm_const_tensor
            fc1_glu_kwargs["discrete_col_sfd"] = not use_nvfp4

        if fc1_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM.
            fc1_weight_for_gemm = grouped_fc1_weight.copy()
            tex.grouped_swizzle_for_gemm(fc1_weight_for_gemm, rowwise=True, columnwise=False)

            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, n, k)
            # Data logical shape: (n, k, num_groups)
            fc1_w_data = fc1_weight_for_gemm.rowwise_data
            fc1_w_data = fc1_w_data.view(dtype=data_dtype)
            fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], fc1_weight_k)
            fc1_w_data = fc1_w_data.permute(1, 2, 0)
            fc1_w_scales = fc1_weight_for_gemm.scale_inv.view(dtype=scale_view_dtype)
            fc1_w_scales = fc1_w_scales.view(
                num_groups,
                (fc1_weight_shape[0] + 127) // 128,
                (fc1_weight_shape[1] + k_sf_divisor - 1) // k_sf_divisor,
                32,
                4,
                4,
            )
            fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

            fc1_glu_kwargs["b_tensor"] = fc1_w_data
            fc1_glu_kwargs["sfb_tensor"] = fc1_w_scales
        else:
            # Discrete-weight kernel: per-expert data/scale pointers
            fc1_weight_data_for_ptrs = [w._rowwise_data for w in grouped_fc1_weight]
            if use_nvfp4:
                fc1_weight_data_for_ptrs = [
                    _nvfp4_logical_data_view(data) for data in fc1_weight_data_for_ptrs
                ]
            fc1_b_ptrs, fc1_sfb_ptrs, _fc1_sw = tex.get_device_pointer_for_data_and_scales(
                fc1_weight_data_for_ptrs,
                [w._rowwise_scale_inv for w in grouped_fc1_weight],
                swizzle=True,
                rowwise=True,
                data_dtype=(
                    grouped_fc1_weight[0]._fp4_dtype
                    if use_nvfp4
                    else grouped_fc1_weight[0]._fp8_dtype
                ),
            )
            fc1_glu_kwargs["b_ptrs"] = fc1_b_ptrs
            fc1_glu_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
            fc1_glu_kwargs["n"] = fc1_weight_shape[0]
            fc1_glu_kwargs["b_dtype"] = data_dtype
            fc1_glu_kwargs["b_major"] = "k"

        if use_fc1_glu_hadamard:
            fc1_kernel_out = fc1_glu_hadamard_kernel(**fc1_glu_kwargs)
        else:
            fc1_kernel_out = self.grouped_gemm_glu_kernel()(**fc1_glu_kwargs)

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m_splits), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m_splits)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (sum(m_splits), k, 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m_splits)/128, 1)
        swiglu_in = fc1_kernel_out["c_tensor"]
        swiglu_in = swiglu_in.view(in_shape[0], fc1_weight_shape[0])

        if use_nvfp4:
            # The NVFP4 fused dGLU path emits unquantized BF16 ``d_tensor``;
            # re-quantize to NVFP4 to feed the FC2 grouped GEMM and to provide
            # a columnwise tile for FC2 wgrad.
            fc2_in = fc1_kernel_out["d_tensor"]
            fc2_in = fc2_in.view(in_shape[0], fc2_weight_shape[1])
            _debug_grouped_mlp_tensor(
                "fc1_post_glu_d_tensor",
                fc2_in,
                prefix=(
                    "[TEFusedGroupedMLPDebug "
                    f"fc1_glu_rht_amax={int(use_fc1_glu_hadamard)} "
                    f"num_groups={num_groups}]"
                ),
            )
            fc2_in = fc2_in.contiguous()
            fc2_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            fc2_input_quantizer.optimize_for_gemm = True
            _enable_nvfp4_rht_for_group_quantize(fc2_input_quantizer)
            if use_fc1_glu_hadamard:
                grouped_fc2_x = _group_quantize_with_amax_for_grouped_mlp(
                    fc2_in,
                    fc2_input_quantizer,
                    num_groups,
                    split_sizes,
                    fc1_kernel_out["amax_tensor"].view(-1),
                    fc1_kernel_out["post_rht_amax_tensor"].view(-1),
                    tensor_offsets=fc2_x_tensor_offsets,
                )
            else:
                grouped_fc2_x = _group_quantize_for_grouped_mlp(
                    fc2_in,
                    fc2_input_quantizer,
                    num_groups,
                    split_sizes,
                    tensor_offsets=fc2_x_tensor_offsets,
                )
        else:
            fc2_in_row_data = fc1_kernel_out["d_tensor"]
            fc2_in_row_data = fc2_in_row_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_row_scale = fc1_kernel_out["sfd_row_tensor"]
            fc2_in_row_scale = fc2_in_row_scale.permute(5, 2, 4, 0, 1, 3)

            fc2_in_col_data = fc1_kernel_out["d_col_tensor"]
            fc2_in_col_data = fc2_in_col_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_col_scale = fc1_kernel_out["sfd_col_tensor"]
            fc2_in_col_scale = fc2_in_col_scale.permute(5, 2, 4, 0, 1, 3)
            # Repack columnwise scales on GPU to preserve group ordering.

            # FC2 inputs scales are already swizzled/optimized for GEMM
            grouped_fc2_x = GroupedTensor(
                shape=(in_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_input_quantizer,
                data=fc2_in_row_data.reshape(-1),
                columnwise_data=fc2_in_col_data.reshape(-1),
                scale_inv=fc2_in_row_scale.reshape(-1),
                columnwise_scale_inv=fc2_in_col_scale.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_weight_shape[0]]
        fc2_scales = basic_op_extra_inputs[2][1] if fc2_op._scale_bias else None

        if use_nvfp4:
            # NVFP4 GEMM uses the generic grouped GEMM wrapper which handles
            # quantized GroupedTensor inputs end-to-end (data + swizzled scales).
            # Bias / bias scaling are applied as a separate elementwise op below
            # to avoid the cuDNN-specific packed bias layout.
            fc2_out_buf = torch.empty(fc2_out_shape, dtype=dtype, device=device)
            if (
                num_groups == 1
                and grouped_fc2_x.columnwise_data is not None
                and grouped_fc2_x.columnwise_scale_inv is not None
            ):
                if fc2_op.single_grouped_weight:
                    fc2_w_single = grouped_fc2_weight.split_into_quantized_tensors()[0]
                else:
                    fc2_w_single = grouped_fc2_weight[0]

                m_fc2, k_fc2 = grouped_fc2_x.logical_shape
                fc2_x_single_shape = (m_fc2, k_fc2)
                fc2_x_single = NVFP4Tensor(
                    shape=fc2_x_single_shape,
                    dtype=dtype,
                    rowwise_data=grouped_fc2_x.rowwise_data.view(
                        fc2_input_quantizer.convert_shape_for_fp4(fc2_x_single_shape)
                    ),
                    rowwise_scale_inv=grouped_fc2_x.scale_inv.view(
                        fc2_input_quantizer.get_scale_shape(fc2_x_single_shape, False)
                    ),
                    columnwise_data=grouped_fc2_x.columnwise_data.view(
                        fc2_input_quantizer.convert_shape_for_fp4(
                            fc2_input_quantizer.get_columnwise_shape(fc2_x_single_shape)
                        )
                    ),
                    columnwise_scale_inv=grouped_fc2_x.columnwise_scale_inv.view(
                        fc2_input_quantizer.get_scale_shape(fc2_x_single_shape, True)
                    ),
                    amax_rowwise=grouped_fc2_x.amax,
                    amax_columnwise=grouped_fc2_x.columnwise_amax,
                    fp4_dtype=getattr(fc2_w_single, "_fp4_dtype", fc2_input_quantizer.dtype),
                    quantizer=fc2_input_quantizer,
                    requires_grad=False,
                    with_gemm_swizzled_scales=getattr(
                        grouped_fc2_x,
                        "_with_gemm_swizzled_scales",
                        getattr(grouped_fc2_x, "with_gemm_swizzled_scales", True),
                    ),
                )
                general_gemm(
                    fc2_w_single,
                    fc2_x_single,
                    out_dtype=dtype,
                    out=fc2_out_buf,
                    layout="TN",
                    use_split_accumulator=False,
                )
            else:
                fc2_out_offsets = base_offsets * fc2_weight_shape[0]
                fc2_out_grouped = GroupedTensor(
                    shape=(in_shape[0], fc2_weight_shape[0]),
                    dtype=dtype,
                    num_tensors=num_groups,
                    quantizer=None,
                    data=fc2_out_buf.view(-1),
                    first_dims=split_sizes,
                    tensor_offsets=fc2_out_offsets,
                )
                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc2_weight,
                    grouped_fc2_x,
                    fc2_out_grouped,
                    layout="TN",
                )
            fc2_out = fc2_out_buf
            if fc2_bias_packed is not None:
                # ``fc2_bias_packed`` has shape (n, num_groups) with stride (1, n)
                # for cuDNN. For NVFP4 we apply bias per-token using the saved
                # split sizes.
                bias_per_group = fc2_bias_packed.transpose(0, 1).contiguous()
                token_bias = torch.repeat_interleave(bias_per_group, split_sizes, dim=0)
                if fc2_scales is not None:
                    fc2_out = fc2_out + token_bias * fc2_scales.view(-1, 1)
                else:
                    fc2_out = fc2_out + token_bias
        else:
            fc2_scales_tensor = (
                fc2_scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1)
                if fc2_scales is not None
                else torch.ones((in_shape[0], 1, 1), dtype=torch.float32, device=device)
            )
            fc2_quant_kwargs = {
                "a_tensor": fc1_kernel_out["d_tensor"],
                "sfa_tensor": fc1_kernel_out["sfd_row_tensor"],
                "padded_offsets": split_points,
                "alpha_tensor": alpha_tensor.float(),
                "norm_const_tensor": None,
                "prob_tensor": fc2_scales_tensor,
                "acc_dtype": torch.float32,
                "d_dtype": dtype,
                "cd_major": "n",
                "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                "current_stream": current_stream,
                "use_dynamic_sched": True,
            }
            if self.is_fc2_bias_supported():
                fc2_quant_kwargs["bias_tensor"] = fc2_bias_packed

            if fc2_op.single_grouped_weight:
                # Clone and swizzle scales for GEMM (original stays unmodified for save_for_backward)
                fc2_weight_for_gemm = grouped_fc2_weight.copy()
                tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=True, columnwise=False)

                fc2_w_data = fc2_weight_for_gemm.rowwise_data
                fc2_w_data = fc2_w_data.view(dtype=torch.float8_e4m3fn)
                fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_shape[1])
                fc2_w_data = fc2_w_data.permute(1, 2, 0)

                fc2_w_scales = fc2_weight_for_gemm.scale_inv.view(dtype=torch.float8_e8m0fnu)
                fc2_w_scales = fc2_w_scales.view(
                    num_groups,
                    (fc2_weight_shape[0] + 127) // 128,
                    (fc2_weight_shape[1] + 127) // 128,
                    MXFP8_BLOCK_SCALING_SIZE,
                    4,
                    4,
                )
                fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)
                fc2_quant_kwargs["b_tensor"] = fc2_w_data
                fc2_quant_kwargs["sfb_tensor"] = fc2_w_scales
            else:
                fc2_b_ptrs, fc2_sfb_ptrs, _ = tex.get_device_pointer_for_data_and_scales(
                    [w._rowwise_data for w in grouped_fc2_weight],
                    [w._rowwise_scale_inv for w in grouped_fc2_weight],
                    swizzle=True,
                    rowwise=True,
                    data_dtype=grouped_fc2_weight[0]._fp8_dtype,
                )
                fc2_quant_kwargs["b_ptrs"] = fc2_b_ptrs
                fc2_quant_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
                fc2_quant_kwargs["n"] = fc2_weight_shape[0]
                fc2_quant_kwargs["b_dtype"] = torch.float8_e4m3fn
                fc2_quant_kwargs["b_major"] = "k"

            fc2_kernel_out = self.grouped_gemm_quant_kernel()(**fc2_quant_kwargs)
            fc2_out = fc2_kernel_out["d_tensor"].permute(2, 0, 1).view(fc2_out_shape).contiguous()

        # Save state for backward pass
        if requires_grad:
            mark_grouped_tensor(grouped_fc1_x, swiglu_in, scales, grouped_fc2_x)
            fc1_input_tensors = (
                grouped_fc1_x.rowwise_data,
                grouped_fc1_x.columnwise_data,
                grouped_fc1_x.scale_inv,
                grouped_fc1_x.columnwise_scale_inv,
                fc1_x_tensor_offsets,
                grouped_fc1_x.amax if use_nvfp4 else None,
                grouped_fc1_x.columnwise_amax if use_nvfp4 else None,
            )
            # FC1
            fc1_weight_tensors = (
                [grouped_fc1_weight] if fc1_op.single_grouped_weight else grouped_fc1_weight
            )
            fc1_ctx.save_for_backward(
                split_sizes, split_points, *fc1_weight_tensors, *fc1_input_tensors
            )
            fc1_ctx.with_quantized_compute = True
            fc1_ctx.input_quantizer = fc1_input_quantizer
            fc1_ctx.weight_quantizer = fc1_weight_quantizer
            fc1_ctx.grad_output_quantizer = fc1_grad_output_quantizer
            fc1_ctx.grad_output_quantizers = [fc1_grad_output_quantizer]
            fc1_ctx.grad_input_quantizers = None
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = weight_requires_grad
            fc1_ctx.base_split_offsets = base_offsets

            # Scaled SwiGLU
            swiglu_ctx.save_for_backward(swiglu_in, scales)
            swiglu_ctx.input_requires_grad = True
            swiglu_ctx.extra_input_requires_grad = True
            swiglu_ctx.dtype = dtype

            # FC2 state
            if grouped_fc2_x is not None:
                fc2_input_tensors = (
                    grouped_fc2_x.rowwise_data,
                    grouped_fc2_x.columnwise_data,
                    grouped_fc2_x.scale_inv,
                    grouped_fc2_x.columnwise_scale_inv,
                    fc2_x_tensor_offsets,
                    grouped_fc2_x.amax if use_nvfp4 else None,
                    grouped_fc2_x.columnwise_amax if use_nvfp4 else None,
                )
            else:
                fc2_input_tensors = (None, None, None, None, None, None, None)

            if fc2_op.single_grouped_weight:
                fc2_ctx.save_for_backward(split_sizes, grouped_fc2_weight, *fc2_input_tensors)
            else:
                fc2_ctx.save_for_backward(split_sizes, *grouped_fc2_weight, *fc2_input_tensors)

            fc2_ctx.with_quantized_compute = True
            fc2_ctx.input_quantizer = fc2_input_quantizer
            fc2_ctx.weight_quantizer = fc2_weight_quantizer
            fc2_ctx.grad_output_quantizer = fc2_grad_output_quantizer
            fc2_ctx.grad_output_quantizers = [fc2_grad_output_quantizer]
            fc2_ctx.grad_input_quantizers = None
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad

        return fc2_out, [(), (), ()]


def fuse_forward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply operation fusion for forward pass.

    Parameters
    ----------
    ops : list of FusibleOperation
        Forward pass operations.
    recipe : Recipe, optional
        Quantization recipe.

    Returns
    -------
    ops : list of FusibleOperation
        Updated forward pass operations

    """

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8,
    )


# Register fusion if available
if ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported():
    register_forward_fusion(fuse_forward_ops, prepend=True)
