import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.Dequant", ops=["dequant"])
@pytest.mark.parametrize("tokens,hidden", [
    (1, 4096),
    (32, 4096),
    (128, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (2048, 8192),
    (4096, 8192),
    (8192, 8192),
])
def test_dequant(benchmark, perf_environment, tokens, hidden):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.Dequant(output_dtype=torch.bfloat16, implementation=implementation)
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
        scale = torch.rand(hidden, dtype=torch.float32, device=device)
        return lambda: module(x, scale)

    benchmark(
        factory=factory,
        op="dequant",
        tokens=tokens,
        hidden=hidden,
        dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("modules.DynamicQuant", ops=["dynamic_quant"])
@pytest.mark.parametrize("tokens,hidden", [
    (1, 4096),
    (32, 4096),
    (128, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (2048, 8192),
    (4096, 8192),
    (8192, 8192),
])
def test_dynamic_quant(benchmark, perf_environment, tokens, hidden):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.DynamicQuant(implementation=implementation)
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
        return lambda: module(x)

    benchmark(
        factory=factory,
        op="dynamic_quant",
        tokens=tokens,
        hidden=hidden,
        dtype="bfloat16",
        phase="forward",
    )
