import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data  = {
    'case1': {
        'data': torch.randn(32, 8).to(torch.float16),  # 32个元素
        'reduce': 'sum',
        'lengths': torch.tensor([10, 12, 10]).to(torch.int32),  # 10+12+10=32
        'axis': 0,
        'initial': torch.tensor(0.0).to(torch.bfloat16),  # sum操作的合理初始值
        'grad_output': torch.randn(3, 8).to(torch.float16) * 0.1
    }
}

def verify_result(real_result, golden):
    # 根据数据类型设置误差阈值（与原逻辑一致）
    if golden.dtype == torch.float32:
        rtol = 1e-4  # fp32相对误差阈值
        atol = 1e-4  # fp32绝对误差阈值
    else:
        rtol = 1e-3  # fp16相对误差阈值
        atol = 1e-3  # fp16绝对误差阈值

    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)

    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    # 补充NaN判断（与equal_nan=True保持一致）
    both_nan = torch.isnan(real_result) & torch.isnan(golden)
    is_close = is_close | both_nan
    err_num = torch.sum(~is_close).item()  # 取反统计不满足的数量

    # 与原逻辑相同的误差数量判断
    if real_result.numel() * rtol < err_num:
        print(f"[ERROR] result error")
        return False
    print("test pass")
    return True

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        case_name = 'case' + str(num)
        
        # 获取测试数据
        data = case_data[case_name]["data"]
        grad_output = case_data[case_name]["grad_output"]
        lengths = case_data[case_name].get("lengths", None)
        offsets = case_data[case_name].get("offsets", None)
        initial = case_data[case_name].get("initial", None)
        reduce = case_data[case_name]["reduce"]
        axis = case_data[case_name]["axis"]

        data.requires_grad_(True)
        if lengths is not None:
            output = torch.segment_reduce(data, reduce=reduce, lengths=lengths,initial=initial, axis=axis)
        elif offsets is not None:
            output = torch.segment_reduce(data, reduce=reduce, offsets=offsets,initial=initial, axis=axis)

        output.backward(grad_output)
        pytorch_grad = data.grad.clone()

        lengths_npu = lengths.npu() if lengths is not None else None
        offsets_npu = offsets.npu() if offsets is not None else None
        initial_npu = initial.npu() if initial is not None else None

        output_npu= custom_ops_lib.custom_op(grad_output.npu(),output.detach().npu(),data.detach().npu(),lengths_npu,offsets_npu,initial_npu, reduce, axis, pytorch_grad)
        if output_npu is None:
            print(f"{case_name} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), pytorch_grad):
                print(f"{case_name} verify result pass!")
            else:
                print(f"{case_name} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
