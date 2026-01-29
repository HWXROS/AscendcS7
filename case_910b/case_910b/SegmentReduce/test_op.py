import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'data': torch.rand(32, dtype=torch.float16),
        'lengths': torch.tensor([10, 12, 10]),
        'reduce': 'sum',
        'axis': 0,
        'unsafe': False
    }
}

def verify_result(real_result, golden, flag):
    # 根据数据类型设置误差阈值（与原逻辑一致）
    if golden.dtype == torch.float32:
        rtol = 1e-4  # fp32相对误差阈值
        atol = 1e-4  # fp32绝对误差阈值
    elif flag == True:
        rtol = 1e-2  # fp16相对误差阈值
        atol = 1e-2  # fp16绝对误差阈值
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
        input_data = case_data[case_name]["data"]
        input_lengths = case_data[case_name].get("lengths", None)
        input_offsets = case_data[case_name].get("offsets", None)
        input_initial = case_data[case_name].get("initial", None)
        reduce = case_data[case_name]["reduce"]
        axis = case_data[case_name]["axis"]
        unsafe = case_data[case_name]["unsafe"]

        output = torch.segment_reduce(input_data, lengths=input_lengths, offsets=input_offsets, initial=input_initial, reduce=reduce, axis=axis,  unsafe=unsafe)
        
        input_lengths_npu = input_lengths.npu() if input_lengths is not None else None
        input_offsets_npu = input_offsets.npu() if input_offsets is not None else None
        input_initial_npu = input_initial.npu() if input_initial is not None else None

        output_npu= custom_ops_lib.custom_op(input_data.npu(), input_lengths_npu, input_offsets_npu, input_initial_npu, reduce, axis, unsafe, output)
        if output_npu is None:
            print(f"{case_name} execution timed out!")
        else:
            if reduce in ('sum', 'mean', 'prod'):
                low_precision = True
            else:
                low_precision = False
            if verify_result(output_npu.cpu(), output, low_precision):
                print(f"{case_name} verify result pass!")
            else:
                print(f"{case_name} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
