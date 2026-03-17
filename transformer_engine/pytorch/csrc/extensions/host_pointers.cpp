/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <ATen/cuda/CUDAContext.h>

namespace transformer_engine::pytorch {

std::vector<at::Tensor> convert_host_pointers_to_tensor(
    std::vector<std::vector<at::Tensor>> tensor_lists) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensor_lists.size());

  for (const auto& tensor_list : tensor_lists) {
    NVTE_CHECK(!tensor_list.empty(), "Tensor list is empty.");
    const auto& first_tensor = tensor_list[0];
    NVTE_CHECK(first_tensor.is_cuda(), "Tensor list must be on CUDA.");
    const auto device = first_tensor.device();

    const int64_t count = static_cast<int64_t>(tensor_list.size());

    // Collect device pointers on host
    std::vector<int64_t> host_ptrs(count);
    for (int64_t i = 0; i < count; ++i) {
      host_ptrs[i] = static_cast<int64_t>(
          reinterpret_cast<uintptr_t>(tensor_list[static_cast<size_t>(i)].data_ptr()));
    }

    // Allocate device output and launch kernel
    auto options = at::TensorOptions().dtype(at::kLong).device(device);
    auto out = at::empty({count}, options);
    auto stream = at::cuda::getCurrentCUDAStream();
    nvte_convert_pointers_to_tensor(host_ptrs.data(), out.data_ptr<int64_t>(), count, stream);
    outputs.push_back(out);
  }

  return outputs;
}

}  // namespace transformer_engine::pytorch
