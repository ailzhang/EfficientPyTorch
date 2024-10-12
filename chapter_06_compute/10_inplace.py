import torch
from torch.profiler import profile, ProfilerActivity


def run(data, use_inplace):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(2):
            if use_inplace:
                data.mul_(2)
            else:
                output = data.mul(2)
    prof.export_chrome_trace(f"traces/PROF_use_inplace_{use_inplace}.json")


shape = (32, 32, 256, 256)

# Non-Inplace
data1 = torch.randn(shape, device="cuda:0")
run(data1, use_inplace=False)

# Inplace
data2 = torch.randn(shape, device="cuda:0")
run(data2, use_inplace=True)
