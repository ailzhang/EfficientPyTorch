import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1000, 5000)
        self.linear2 = nn.Linear(5000, 10000)
        self.linear3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.linear1(x))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        nonzero = torch.nonzero(output)
        return nonzero


def run(data, model):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            model(data)
    prof.export_chrome_trace("traces/PROF_nonzero.json")


data = torch.randn(1, 1000, device="cuda")
model = Model().to(torch.device("cuda"))
run(data, model)
