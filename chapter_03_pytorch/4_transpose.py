import torch

# 创建一个3*4的张量, 使用contiguous()确保其连续性
x = torch.arange(12).reshape(3, 4).contiguous()

print(f"x = {x}\nx.stride = {x.stride()}")
# x = tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# x.stride = (4, 1)

y = torch.as_strided(x, size=(4, 3), stride=(1, 4))
print(f"y = {y}\ny.stride = {y.stride()}")
# y = tensor([[ 0,  4,  8],
#         [ 1,  5,  9],
#         [ 2,  6, 10],
#         [ 3,  7, 11]])
# y.stride = (1, 4)

# 张量x和y共享同一块底层存储
assert id(x.untyped_storage()) == id(y.untyped_storage())
