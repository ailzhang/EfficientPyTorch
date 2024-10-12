import torch

# 创建一个10*20的张量, 使用contiguous()确保其连续性
x = torch.arange(200).reshape(10, 20).contiguous()

# 基础索引，读取x的第0行
y_basic_index = x[0]

# (1) 基于基础索引进行读取的返回张量和x共享底层存储
assert y_basic_index.data_ptr() == x.data_ptr()

# 使用整数张量对x进行高级索引，返回位置在[0, 2], [1, 3], [2, 4]位置的元素
z_adv_index_int = x[torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])]
# z_adv_index_int = tensor([ 2, 23, 44])

# 对张量x中的每个元素进行判断，如果元素的值小于10，则对应位置的ind为True，否则为False
ind = x < 10
# 使用布尔张量对x进行高级索引，返回x中所有对应ind位置为True的元素
z_adv_index_bool = x[ind]
# z_adv_index_bool = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# (2) 基于高级索引进行读取的返回张量和x的底层存储是分开的
assert z_adv_index_int.data_ptr() != x.data_ptr()
assert z_adv_index_bool.data_ptr() != x.data_ptr()
