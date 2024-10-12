import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


def tensor_creation(num_iters, create_on_gpu):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        shape = (10, 6400)
        for i in range(num_iters):
            if create_on_gpu:
                data = torch.randn(shape, device="cuda")
            else:
                data = torch.randn(shape).to("cuda")
    prof.export_chrome_trace(
        f"traces/PROF_tensor_creation_on_gpu_{create_on_gpu}.json"
    )


# 情况1. 先在CPU上创建Tensor然后拷贝到GPU
tensor_creation(20, create_on_gpu=False)

# 情况2. 直接在GPU上创建Tensor
tensor_creation(20, create_on_gpu=True)
