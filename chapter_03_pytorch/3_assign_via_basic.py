import torch

# 创建一个10*20的张量, 使用contiguous()确保其连续性
x = torch.arange(200).reshape(10, 20).contiguous()

# 通过基础索引对x的[0, 0]元素进行赋值
x[0, 0] = -1.0
print(x[0, 0])  # x[0, 0]被更新成-1.0

# 通过切片索引对x[2, :]的所有元素进行赋值
x[2, :] = 10
print(x)  # x的第2行(从0计数）的所有元素被更新成10
