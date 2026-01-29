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


at::Tensor my_op_impl_npu(const at::Tensor& grad, const at::Tensor& output, const at::Tensor& data, const c10::optional<at::Tensor>& lengths, const c10::optional<at::Tensor>& offsets, const c10::optional<at::Tensor>& initial, const std::string reduce, int64_t axis, const at::Tensor& golden_output) {
     at::Tensor result ;
    const char* reduce_cstr = reduce.c_str();
    auto round = 50 ;
    for (size_t i = 0; i < round; i++)
    {
        result = at::empty(golden_output.sizes(), golden_output.options().device(data.device()));
        EXEC_NPU_CMD(aclnnSegmentReduceGrad, grad, output, data, lengths, offsets, initial, reduce_cstr, axis, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor grad,Tensor output, Tensor data, Tensor? lengths, Tensor? offsets, Tensor? initial, str reduce, int axis, Tensor golden_output) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch segment_reduce_backward");
}
