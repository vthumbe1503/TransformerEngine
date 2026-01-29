/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <transformer_engine/cast.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/recipe.h>

#include "../common.h"
#include "../recipe/recipe_common.cuh"
#include "../transpose/cast_transpose.h"
#include "../util/multi_stream.h"
#include "../utils.cuh"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"
#include "dispatch/quantize_grouped.cuh"
#include "transformer_engine/transpose.h"

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
}

void nvte_quantize_grouped(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_grouped);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_grouped_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
}

namespace {

constexpr int kGroupedQuantizeThreads = 256;
constexpr int kGroupedElementsPerThread = 4;

__launch_bounds__(kGroupedQuantizeThreads) __global__
    void zero_grouped_amax_kernel(float *amax, const size_t num_tensors, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tensors) {
    amax[idx] = 0.0f;
  }
}

template <typename IType>
__launch_bounds__(kGroupedQuantizeThreads) __global__
    void amax_grouped_current_scaling_kernel(
        const IType *input, float *amax, const int64_t *offsets, const int64_t *first_dims,
        const int64_t *last_dims, const size_t num_tensors, const bool all_same_first,
        const bool all_same_last, const bool all_same_shape, const size_t common_first,
        const size_t common_last, const int blocks_per_tensor, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const int tensor_idx = blockIdx.x / blocks_per_tensor;
  const int block_in_tensor = blockIdx.x % blocks_per_tensor;
  if (tensor_idx >= static_cast<int>(num_tensors)) {
    return;
  }

  const size_t m = all_same_first ? common_first : static_cast<size_t>(first_dims[tensor_idx]);
  const size_t n = all_same_last ? common_last : static_cast<size_t>(last_dims[tensor_idx]);
  const size_t numel = m * n;
  if (numel == 0) {
    return;
  }

  const size_t offset =
      all_same_shape ? (static_cast<size_t>(tensor_idx) * numel)
                     : static_cast<size_t>(offsets[tensor_idx]);
  const size_t elements_per_block = blockDim.x * kGroupedElementsPerThread;
  const size_t start = static_cast<size_t>(block_in_tensor) * elements_per_block;
  if (start >= numel) {
    return;
  }
  const size_t end = min(start + elements_per_block, numel);

  float local_max = 0.0f;
  for (size_t idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
    const float val = static_cast<float>(input[offset + idx]);
    local_max = fmaxf(local_max, fabsf(val));
  }

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  local_max =
      transformer_engine::reduce_max<kGroupedQuantizeThreads / THREADS_PER_WARP>(local_max, warp_id);
  if (threadIdx.x == 0) {
    transformer_engine::atomicMaxFloat(&amax[tensor_idx], local_max);
  }
}

template <typename OType>
__launch_bounds__(kGroupedQuantizeThreads) __global__
    void compute_grouped_scale_kernel(const float *amax_ptr, float *scale_ptr, float *scale_inv_ptr,
                                      float *columnwise_scale_inv_ptr, const size_t num_tensors,
                                      const bool force_pow_2_scales, const float amax_epsilon,
                                      const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tensors) {
    return;
  }
  const float amax = amax_ptr[idx];
  constexpr float max_fp8 = transformer_engine::Quantized_Limits<OType>::max_norm;
  const float scale = transformer_engine::compute_scale_from_amax(
      amax, max_fp8, force_pow_2_scales, amax_epsilon, std::numeric_limits<float>::max());
  scale_ptr[idx] = scale;
  scale_inv_ptr[idx] = 1.0f / scale;
  if (columnwise_scale_inv_ptr != nullptr) {
    columnwise_scale_inv_ptr[idx] = scale_inv_ptr[idx];
  }
}

template <typename IType, typename OType>
__launch_bounds__(kGroupedQuantizeThreads) __global__
    void quantize_grouped_current_scaling_kernel(
        const IType *input, OType *output, OType *output_colwise, const float *scale,
        const int64_t *offsets, const int64_t *first_dims, const int64_t *last_dims,
        const size_t num_tensors, const bool all_same_first, const bool all_same_last,
        const bool all_same_shape, const size_t common_first, const size_t common_last,
        const int blocks_per_tensor, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const int tensor_idx = blockIdx.x / blocks_per_tensor;
  const int block_in_tensor = blockIdx.x % blocks_per_tensor;
  if (tensor_idx >= static_cast<int>(num_tensors)) {
    return;
  }

  const size_t m = all_same_first ? common_first : static_cast<size_t>(first_dims[tensor_idx]);
  const size_t n = all_same_last ? common_last : static_cast<size_t>(last_dims[tensor_idx]);
  const size_t numel = m * n;
  if (numel == 0) {
    return;
  }

  const size_t offset =
      all_same_shape ? (static_cast<size_t>(tensor_idx) * numel)
                     : static_cast<size_t>(offsets[tensor_idx]);
  const size_t elements_per_block = blockDim.x * kGroupedElementsPerThread;
  const size_t start = static_cast<size_t>(block_in_tensor) * elements_per_block;
  if (start >= numel) {
    return;
  }
  const size_t end = min(start + elements_per_block, numel);

  const float scale_val = scale[tensor_idx];
  for (size_t idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
    const float val = static_cast<float>(input[offset + idx]) * scale_val;
    output[offset + idx] = static_cast<OType>(val);
    if (output_colwise != nullptr) {
      const size_t row = idx / n;
      const size_t col = idx % n;
      output_colwise[offset + col * m + row] = static_cast<OType>(val);
    }
  }
}

}  // namespace

void nvte_quantize_grouped_current_scaling(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                           const NVTEQuantizationConfig quant_config,
                                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_grouped_current_scaling);
  using namespace transformer_engine;

  NVTE_CHECK(quant_config != nullptr,
             "Quantization config must be provided for grouped current scaling.");
  const GroupedTensor *input_tensor = convertNVTEGroupedTensorCheck(input);
  GroupedTensor *output_tensor = convertNVTEGroupedTensorCheck(output);

  NVTE_CHECK(input_tensor->num_tensors == output_tensor->num_tensors,
             "Input and output must have the same number of tensors.");
  NVTE_CHECK(input_tensor->logical_shape.ndim == output_tensor->logical_shape.ndim,
             "Input and output grouped tensors must have the same logical shape rank.");
  for (size_t i = 0; i < output_tensor->logical_shape.ndim; ++i) {
    NVTE_CHECK(input_tensor->logical_shape.data[i] == output_tensor->logical_shape.data[i],
               "Input and output grouped tensors must have the same logical shape.");
  }
  NVTE_CHECK(input_tensor->has_data(), "Input grouped tensor must have rowwise data.");
  NVTE_CHECK(output_tensor->has_data(),
             "Output grouped tensor must have rowwise data for current scaling.");
  NVTE_CHECK(output_tensor->scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Grouped current scaling expects NVTE_DELAYED_TENSOR_SCALING output, got ",
             to_string(output_tensor->scaling_mode));
  NVTE_CHECK(output_tensor->scale.dptr != nullptr, "Output grouped tensor must have scale buffer.");
  NVTE_CHECK(output_tensor->amax.dptr != nullptr, "Output grouped tensor must have amax buffer.");
  NVTE_CHECK(output_tensor->scale_inv.dptr != nullptr,
             "Output grouped tensor must have scale_inv buffer.");
  if (output_tensor->has_columnwise_data()) {
    NVTE_CHECK(output_tensor->columnwise_scale_inv.dptr != nullptr,
               "Output grouped tensor must have columnwise_scale_inv buffer.");
  }
  if (output_tensor->has_columnwise_data()) {
    NVTE_CHECK(output_tensor->has_data(),
               "Columnwise-only grouped current scaling is not supported.");
  }

  QuantizationConfig quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  const float *noop_ptr = nullptr;
  if (quant_config_cpp.noop_tensor != nullptr) {
    auto *noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
    noop_ptr = reinterpret_cast<const float *>(noop_tensor->data.dptr);
  }

  const size_t num_tensors = output_tensor->num_tensors;

  const bool all_same_first = output_tensor->all_same_first_dim();
  const bool all_same_last = output_tensor->all_same_last_dim();
  const bool all_same_shape = output_tensor->all_same_shape();
  const size_t common_first = all_same_first ? output_tensor->get_common_first_dim() : 0;
  const size_t common_last = all_same_last ? output_tensor->get_common_last_dim() : 0;

  size_t max_numel = 0;
  if (all_same_first && all_same_last) {
    max_numel = common_first * common_last;
  }

  std::vector<int64_t> first_dims_host;
  std::vector<int64_t> last_dims_host;
  if (!all_same_first) {
    NVTE_CHECK(output_tensor->first_dims.dptr != nullptr, "first_dims is not allocated.");
    first_dims_host.resize(num_tensors);
    NVTE_CHECK_CUDA(cudaMemcpy(first_dims_host.data(), output_tensor->first_dims.dptr,
                               sizeof(int64_t) * num_tensors, cudaMemcpyDeviceToHost));
  }
  if (!all_same_last) {
    NVTE_CHECK(output_tensor->last_dims.dptr != nullptr, "last_dims is not allocated.");
    last_dims_host.resize(num_tensors);
    NVTE_CHECK_CUDA(cudaMemcpy(last_dims_host.data(), output_tensor->last_dims.dptr,
                               sizeof(int64_t) * num_tensors, cudaMemcpyDeviceToHost));
  }
  if (!all_same_shape) {
    NVTE_CHECK(output_tensor->tensor_offsets.dptr != nullptr, "tensor_offsets is not allocated.");
  }

  if (!(all_same_first && all_same_last)) {
    for (size_t i = 0; i < num_tensors; ++i) {
      const size_t m =
          all_same_first ? common_first : static_cast<size_t>(first_dims_host[i]);
      const size_t n = all_same_last ? common_last : static_cast<size_t>(last_dims_host[i]);
      max_numel = std::max(max_numel, m * n);
    }
  }

  if (max_numel == 0) {
    return;
  }

  const auto *input_ptr = reinterpret_cast<const char *>(input_tensor->data.dptr);
  auto *output_ptr = reinterpret_cast<char *>(output_tensor->data.dptr);
  auto *output_col_ptr = output_tensor->has_columnwise_data()
                             ? reinterpret_cast<char *>(output_tensor->columnwise_data.dptr)
                             : nullptr;
  auto *offsets_ptr = all_same_shape ? nullptr
                                     : reinterpret_cast<const int64_t *>(
                                           output_tensor->tensor_offsets.dptr);
  auto *first_dims_ptr =
      all_same_first ? nullptr : reinterpret_cast<const int64_t *>(output_tensor->first_dims.dptr);
  auto *last_dims_ptr =
      all_same_last ? nullptr : reinterpret_cast<const int64_t *>(output_tensor->last_dims.dptr);

  const size_t elems_per_block = kGroupedQuantizeThreads * kGroupedElementsPerThread;
  const int blocks_per_tensor = static_cast<int>(DIVUP(max_numel, elems_per_block));
  const dim3 block(kGroupedQuantizeThreads);
  const dim3 grid_amax(num_tensors * blocks_per_tensor);
  const dim3 grid_scale(DIVUP(num_tensors, static_cast<size_t>(kGroupedQuantizeThreads)));

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input_tensor->data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output_tensor->data.dtype, OType,
          zero_grouped_amax_kernel<<<grid_scale, block, 0, stream>>>(
              reinterpret_cast<float *>(output_tensor->amax.dptr), num_tensors, noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          amax_grouped_current_scaling_kernel<IType><<<grid_amax, block, 0, stream>>>(
              reinterpret_cast<const IType *>(input_ptr),
              reinterpret_cast<float *>(output_tensor->amax.dptr), offsets_ptr, first_dims_ptr,
              last_dims_ptr, num_tensors, all_same_first, all_same_last, all_same_shape,
              common_first, common_last, blocks_per_tensor, noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          compute_grouped_scale_kernel<OType><<<grid_scale, block, 0, stream>>>(
              reinterpret_cast<const float *>(output_tensor->amax.dptr),
              reinterpret_cast<float *>(output_tensor->scale.dptr),
              reinterpret_cast<float *>(output_tensor->scale_inv.dptr),
              output_tensor->has_columnwise_data()
                  ? reinterpret_cast<float *>(output_tensor->columnwise_scale_inv.dptr)
                  : nullptr,
              num_tensors, quant_config_cpp.force_pow_2_scales, quant_config_cpp.amax_epsilon,
              noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          quantize_grouped_current_scaling_kernel<IType, OType><<<grid_amax, block, 0, stream>>>(
              reinterpret_cast<const IType *>(input_ptr),
              reinterpret_cast<OType *>(output_ptr),
              output_col_ptr ? reinterpret_cast<OType *>(output_col_ptr) : nullptr,
              reinterpret_cast<const float *>(output_tensor->scale.dptr), offsets_ptr,
              first_dims_ptr, last_dims_ptr, num_tensors, all_same_first, all_same_last,
              all_same_shape, common_first, common_last, blocks_per_tensor, noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());));
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  // Create config with noop tensor
  QuantizationConfig quant_config;
  quant_config.noop_tensor = noop;

  nvte_quantize_v2(input, output, reinterpret_cast<NVTEQuantizationConfig>(&quant_config), stream);
}

void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, nullptr>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_quantize_dbias_grouped(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                 NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_grouped);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTEGroupedTensor activation_input = nullptr;

  dispatch::quantize_grouped_bwd_helper<IS_DBIAS, IS_DACT, Empty, nullptr>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  dispatch::dequantize_helper(*convertNVTETensorCheck(input), convertNVTETensorCheck(output),
                              stream);
}

void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_configs,
                                const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  const size_t num_streams = nvte_get_num_compute_streams();

  int num_stream_used = std::min(num_streams, num_tensors);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(cudaEventRecord(detail::get_compute_stream_event(0), stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaStreamWaitEvent(detail::get_compute_stream(s), detail::get_compute_stream_event(0)));
  }

  for (int i = 0; i < num_tensors; i++) {
    dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(
        inputs[i], outputs[i], quant_configs, detail::get_compute_stream(i % num_streams));
  }

  // record events on compute streams
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaEventRecord(detail::get_compute_stream_event(s), detail::get_compute_stream(s)));
  }
  // wait for all compute streams to finish
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream, detail::get_compute_stream_event(s)));
  }
}

// Group quantize assumes contiguous inputs and outputs in memory allocation
// TODO (zhongbo): find a better way to make it a more generalized API
void nvte_group_nvfp4_quantize_with_amax(const NVTETensor input, NVTETensor *outputs,
                                         const size_t *split_sections, const size_t num_tensors,
                                         const NVTEQuantizationConfig quant_config,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_nvfp4_quantize_with_amax);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, outputs, split_sections,
                                                              num_tensors, quant_config, stream);
}
