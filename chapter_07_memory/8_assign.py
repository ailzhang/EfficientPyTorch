import torch

shape = [1, 4]
x = torch.ones(shape)
print("Initial x = ", x)  # Initial x =  tensor([[1., 1., 1., 1.]])

y = x
y.mul_(10)

print("Modified y = ", y)  # Modified y =  tensor([[10., 10., 10., 10.]])
print("Modified x = ", x)  # Modified x =  tensor([[10., 10., 10., 10.]])
