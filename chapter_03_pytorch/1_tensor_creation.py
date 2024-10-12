import torch

x = torch.rand((3, 2), dtype=torch.float32, device="cuda")

print(x.dtype)  # torch.float32
print(x.device)  # cuda:0
print(x.shape)  # torch.Size([3, 2])
