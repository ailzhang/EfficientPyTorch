import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


def data_copy(data, dtype_name=""):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            output = data.to("cuda:0", non_blocking=False)
    prof.export_chrome_trace(f"traces/PROF_data_copy_{dtype_name}.json")


# Float precision
data1 = torch.randn(4, 32, 32, 1024, dtype=torch.float32)
data_copy(data1, "float32")


# Uint8 precision
data2 = torch.randint(0, 255, (4, 32, 32, 1024), dtype=torch.uint8)
data_copy(data2, "uint8")
