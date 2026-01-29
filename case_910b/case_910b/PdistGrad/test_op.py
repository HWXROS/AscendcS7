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
        'input': torch.rand(3, 32).uniform_(-1000, 1000).to(torch.float32),
        'p': 2.0,
        'grad': torch.rand(3).uniform_(-1, 1).to(torch.float32)
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
        input_tensor = case_data[case_name]["input"]
        p = case_data[case_name]["p"]
        grad = case_data[case_name]["grad"]
        
        # 计算pdist输出
        pdist_output = F.pdist(input_tensor, p=p)
        
        # 使用PyTorch自动微分计算梯度
        input_tensor.requires_grad_(True)
        output = F.pdist(input_tensor, p=p)
        output.backward(grad)
        pytorch_grad = input_tensor.grad.clone()
        
        output_npu= custom_ops_lib.custom_op(grad.npu(), input_tensor.detach().npu(), pdist_output.npu(), p)
        if output_npu is None:
            print(f"{case_name} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), pytorch_grad):
                print(f"{case_name} verify result pass!")
            else:
                print(f"{case_name} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
