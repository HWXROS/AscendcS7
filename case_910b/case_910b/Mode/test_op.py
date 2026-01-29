import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'input': torch.rand(32).uniform_(0, 10).to(torch.float16),
        'dim': -1,
        'keepdim': False
    }
}

def get_ele(a, dim, keepdim, mode_idx):
    if keepdim:
        elem = torch.gather(a, dim=dim, index=mode_idx)
    else:
        # 补充维度，适配gather的index形状要求
        new_shape = list(mode_idx.shape)
        new_shape.insert(dim, 1)
        mode_idx_expand = mode_idx.reshape(new_shape)
        elem = torch.gather(a, dim=dim, index=mode_idx_expand)
        elem = elem.squeeze(dim=dim)  # 压缩维度，与mode_val形状对齐
    return elem

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
        caseNmae='case'+str(num) 
        input_tensor = case_data[caseNmae]["input"]
        dim = case_data[caseNmae]["dim"]
        keepdim = case_data[caseNmae]["keepdim"]
        
        golden_values, golden_indices = torch.mode(input_tensor, dim=dim, keepdim=keepdim)
        values_npu ,indices_npu= custom_ops_lib.custom_op(input_tensor.npu(), dim, keepdim, golden_values, golden_indices)
        if values_npu is None or indices_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:
            rdim = dim % input_tensor.ndim
            elem = get_ele(input_tensor, rdim, keepdim, indices_npu.cpu())
            if verify_result(values_npu.cpu(), golden_values) and torch.equal(values_npu.cpu(), elem):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    

