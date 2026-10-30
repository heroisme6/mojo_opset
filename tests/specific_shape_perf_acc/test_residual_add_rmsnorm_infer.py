import pytest
import torch

from mojo_opset import modules
from tests.specific_shape_perf_acc.utils import auto_switch_platform


dtypes = [torch.float32, torch.float16, torch.bfloat16]
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", [(16384, 8192), (32768, 8192)])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
def test_residual_add_rms_norm_infer(shape, norm_pos, dtype):
    module = modules.ResidualAddRMSNormInfer(
        shape[-1], eps=1e-5, norm_pos=norm_pos, device="npu", dtype=dtype
    ).requires_grad_(False)
    module.weight.copy_(torch.randn_like(module.weight))
    x = torch.randn(shape, dtype=dtype).npu()
    residual = torch.randn(shape, dtype=dtype).npu()
    perf(lambda: module(x, residual))  # noqa: F821
