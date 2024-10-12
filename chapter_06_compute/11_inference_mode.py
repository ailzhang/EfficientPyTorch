import torch
import torch.nn as nn
import time


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def infer(input_data, num_iters, use_inference_mode):
    start = time.perf_counter()

    with torch.inference_mode(mode=use_inference_mode):
        for _ in range(num_iters):
            output = model(input_data)

    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000


model = SimpleCNN().to(torch.device("cuda:0"))
input_data = torch.randn(1, 3, 224, 224, device="cuda:0")

# 开启Inference Mode
infer(input_data, num_iters=10, use_inference_mode=True)  # warm up
runtime = infer(input_data, num_iters=100, use_inference_mode=True)
print(f"开启Inference Mode用时: {runtime}s")

# 关闭Inference Mode
infer(input_data, num_iters=10, use_inference_mode=False)  # warm up
runtime = infer(input_data, num_iters=100, use_inference_mode=False)
print(f"关闭Inference Mode用时: {runtime}s")
