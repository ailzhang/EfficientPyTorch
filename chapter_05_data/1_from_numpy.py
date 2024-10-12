import numpy as np
import torch

x = np.zeros((3, 3))
y = torch.from_numpy(x)

print(y, type(y))

# tensor([[0., 0., 0.],
#        [0., 0., 0.],
#        [0., 0., 0.]], dtype=torch.float64) <class 'torch.Tensor'>
