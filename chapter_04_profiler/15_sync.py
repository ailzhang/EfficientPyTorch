import time
import torch

sz = 512
N = 10
shape = (sz, sz, sz)

x = torch.randn(dtype=torch.float, size=shape, device="cuda")
y = torch.randn(dtype=torch.float, size=shape, device="cuda")

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(N):
    z = x * y
# 同步
torch.cuda.synchronize()
end = time.perf_counter()
print(f"{N}次运行取平均: {(end - start) / N}s")
