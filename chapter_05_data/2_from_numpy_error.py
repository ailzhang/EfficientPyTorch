import numpy as np
import torch

x = np.random.random(size=(4, 4, 2))
y = np.flip(x, axis=0)

# 报错
# ValueError: At least one stride in the given numpy array is negative,
# and tensors with negative strides are not currently supported.
# (You can probably work around this by making a copy of your array  with array.copy().)
torch.from_numpy(y)

# 创建副本后能够正常运行
torch.from_numpy(y.copy())
