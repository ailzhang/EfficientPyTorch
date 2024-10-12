import torch

x = torch.tensor(2)  # 可以尝试不同的值，如 torch.tensor(1.0)

y = x % 2

if y == 0:
    z = x * 10
else:
    z = x + 10

print(z)
