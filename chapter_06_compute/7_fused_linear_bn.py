import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(100, 50)
        self.bn = nn.BatchNorm1d(50)

    def forward(self, x):
        return self.bn(self.linear(x))


@torch.no_grad()
def run(data, model, num_iters, name):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(num_iters):
            original_output = model(input_tensor)
    prof.export_chrome_trace(f"traces/PROF_cuda_{name}.json")


model = SimpleModel().to(torch.device("cuda:0"))
model.eval()
input_tensor = torch.randn(4, 100, device="cuda:0")

# 融合前
run(input_tensor, model, num_iters=20, name="no_fusion")

# 融合后
fused_model = torch.nn.utils.fusion.fuse_linear_bn_eval(model.linear, model.bn)
run(input_tensor, fused_model, num_iters=20, name="fusion")
