import torch

# 创建一个10*20的张量, 使用contiguous()确保其连续性
x = torch.arange(200).reshape(10, 20).contiguous()

# 对张量x中的每个元素进行判断，如果元素的值小于10，则对应位置的ind为 True，否则为False
ind = x < 10
# 通过高级索引对x的部分元素进行赋值
x[ind] = 1.0

print(x)  # x的对应位置也被更新成1.0
