import torch

a = torch.rand(4, 4)
b = torch.rand(4, 4)
c = torch.rand(4, 4)

x = torch.matmul(a, b)
x1 = x + c
print(x1)


# 融合成一个算子
x2 = torch.addmm(c, a, b)
print(x2)
