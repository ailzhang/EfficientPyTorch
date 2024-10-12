import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
