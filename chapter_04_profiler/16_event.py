import torch

sz = 512
shape = (sz, sz, sz)
x = torch.randn(dtype=torch.float, size=shape, device="cuda")
y = torch.randn(dtype=torch.float, size=shape, device="cuda")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
z = x + y
end.record()

# 等待GPU运行完成
torch.cuda.synchronize()

print(f"用时{start.elapsed_time(end)}ms")
