import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 前向传播：
# 1. 构建并执行前向图
# 2. 构建反向图
t = x * 10
z = t * t

loss = z.mean()

# 反向传播，计算梯度
loss.backward()

# 查看x的梯度
print(x.grad)
