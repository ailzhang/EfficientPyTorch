import torch

a = torch.zeros(3, 3)

# 张量b是张量a的一个视图，共享底层内存
b = a[0]
print(b)  # tensor([0., 0., 0.])

# 修改张量b的内容也会影响张量a
b[0] = 1
print(a)
# tensor([[1., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
