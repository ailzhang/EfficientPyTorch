import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1000, 20000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        y = x
        for _ in range(50):
            y = y * x
        return y


# 未经优化的模型
model = SimpleNet().cuda()

# 打开torch.compile追踪模型的执行过程并自动优化
compiled_model = torch.compile(model)


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


N_ITERS = 5


def benchmark(model):
    times = []
    for i in range(N_ITERS):
        input_data = torch.randn(1000, 1000, device="cuda")
        _, time = timed(lambda: model(input_data))
        times.append(time)
    return times


print("eager模式", benchmark(model))
print("打开torch.compile后", benchmark(compiled_model))

# 输出
# eager模式 [1.1121439208984376, 0.01659187126159668, 0.01635430335998535, 0.016350208282470705, 0.016306175231933593]
# 打开torch.compile后 [1.79336083984375, 0.002367487907409668, 0.0022937600612640383, 0.002292736053466797, 0.002288640022277832]
