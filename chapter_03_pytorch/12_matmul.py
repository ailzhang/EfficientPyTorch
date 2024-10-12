import torch

x1 = torch.rand(32, 32, dtype=torch.float32, device="cuda:0")
x2 = torch.rand(32, 32, dtype=torch.float32, device="cuda:0")

y = x1 @ x2
