/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


std::tuple<at::Tensor, at::Tensor> my_op_impl_npu(const at::Tensor& input, int64_t dim, bool keepdim, const at::Tensor& golden_values, const at::Tensor& golden_indices) {
    // at::Tensor result = at::empty_like(input);
     at::Tensor result_values ;
     at::Tensor result_indices ;
    auto round = 50 ;
    for (size_t i = 0; i < round; i++)
    {
        result_values = at::empty_like(golden_values,at::TensorOptions().dtype(golden_values.dtype()).device(input.options().device()));
        result_indices = at::zeros_like(golden_indices,at::TensorOptions().dtype(golden_indices.dtype()).device(input.options().device()));
        EXEC_NPU_CMD(aclnnMode, input, dim, keepdim, result_values, result_indices);
    }
    return {result_values, result_indices};
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, int dim, bool keepdim, Tensor golden_values, Tensor golden_indices) -> (Tensor, Tensor)");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.mode");
}
