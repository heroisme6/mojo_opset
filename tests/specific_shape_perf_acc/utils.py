"""NPU-only perf helpers mirroring mojo_opset/tests/utils.py of the legacy repo.

`auto_switch_platform(set_perf=True)` moves tensor arguments to the NPU and
injects a module-level `perf` function that profiles kernels with
torch_npu.profiler (Level2, PipeUtilization) and writes the trace into
`./npu_profiling/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/`.
"""

import csv
import functools
import os
import sys
import time

from datetime import datetime

import torch
import torch_npu


def perf(executor, profiling_dir="./npu_profiling", active=5):
    """Profile executor kernels on NPU and report the average kernel duration.

    Returns (avg_kernel_us, profiling_path) or None on failure.
    """

    executor()  # warmup autotune if exists to avoid nested profiler bug
    torch.npu.synchronize()

    os.makedirs(profiling_dir, exist_ok=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        # Baseline matmul, as in the legacy helper.
        mat_a = torch.randn(4096, 4096, dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096, dtype=torch.bfloat16).npu()
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()

        for _ in range(10):
            executor()
            prof.step()
            torch.npu.synchronize()

    try:
        kernel_profiling_path = max(
            [
                os.path.join(profiling_dir, d)
                for d in os.listdir(profiling_dir)
                if os.path.isdir(os.path.join(profiling_dir, d))
            ],
            key=os.path.getmtime,
        )
        csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        if not os.path.exists(csv_file_path):
            print(f"[perf] file not found: {csv_file_path}")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"[perf] failed to locate profiling folder: {exc}")
        return None

    total_durations = 0.0
    count = 0
    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row["Name"].startswith("aclnn"):
                total_durations += float(row["Duration(us)"])
                count += 1
    total_avg_time_us = total_durations / count

    # Host latency for reference.
    for _ in range(3):
        executor()
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        executor()
        torch.npu.synchronize()
    host_latency_ms = (time.time() - start) / 10 * 1000

    print(
        f"[perf] device latency = {total_avg_time_us:.4f} us | host latency = {host_latency_ms:.4f} ms | "
        f"profile dir = {kernel_profiling_path}"
    )
    return total_avg_time_us, kernel_profiling_path


def auto_switch_platform(set_perf: bool = False):
    """Move tensor arguments to the NPU and optionally inject `perf` (NPU only)."""

    if set_perf:
        perf_fn = perf

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [arg.to("npu") if isinstance(arg, torch.Tensor) else arg for arg in args]
            new_kwargs = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            if set_perf:
                setattr(sys.modules[func.__module__], "perf", perf_fn)
            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator
