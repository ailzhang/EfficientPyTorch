import torch
from torch.profiler import profile, ProfilerActivity


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fcs = torch.nn.ModuleList(torch.nn.Linear(200, 200) for i in range(20))

    def forward(self, x):
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        return x


def train(net, optimizer, opt_name=""):
    data = torch.randn(64, 200, device="cuda:0")
    target = torch.randint(0, 1, (64,), device="cuda:0")
    criterion = torch.nn.CrossEntropyLoss()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    prof.export_chrome_trace(f"traces/PROF_perf_{opt_name}.json")


# For-loop
net = SimpleNet().to(torch.device("cuda:0"))
adam_for_loop = torch.optim.Adam(
    net.parameters(), lr=0.01, foreach=False, fused=False
)
train(net, adam_for_loop, opt_name="for_loop")


# For-each
net = SimpleNet().to(torch.device("cuda:0"))
adam_for_each = torch.optim.Adam(
    net.parameters(), lr=0.01, foreach=True, fused=False
)
train(net, adam_for_each, opt_name="for_each")


# Fused
net = SimpleNet().to(torch.device("cuda:0"))
adam_fused = torch.optim.Adam(net.parameters(), lr=0.01, foreach=False, fused=True)
train(net, adam_fused, opt_name="fused")
