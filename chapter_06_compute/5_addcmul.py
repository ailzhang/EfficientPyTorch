import torch

x = torch.rand(3, 3)
y = torch.rand(3, 3)

z = x * y
z1 = z + x
print(z1)

# 可以将上面的计算合并为一个算子，结果是等价的
z2 = torch.addcmul(x, x, y)
print(z2)
