import torch

x = torch.ones((4, 4))

# 原位加法操作
y1 = x.add_(x)
print(y1)  # 张量x所有元素更新为2，张量y1是张量x的一个别名，是同一个张量

# 原位加法操作
x += y1
print(x)  # 张量x所有元素更新为4

# 非原位加法操作
x = x + y1
print(x)  # 张量x所有元素更新为8
