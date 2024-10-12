import torch

x = torch.ones(4, 4)

# torch命名空间下的加法操作
y1 = x.add(x)

# 重载运算符"+"，与x.add(x)等价
y2 = x + x

# Tensor类的加法操作
y3 = torch.add(x, x)

assert (y1 == y2).all()
assert (y2 == y3).all()
