import torch
import torch.optim as optim

shape = [256, 1024, 1024, 1]
weight = torch.randn(shape, requires_grad=True, device="cuda:0")
rand1 = torch.randn(shape, requires_grad=False, device="cuda:0")

x = rand1 * weight
x.sigmoid_()
x.sigmoid_()
x = x.sum()

x.backward()

# 报错信息
# Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
#  [torch.cuda.FloatTensor [256, 1024, 1024, 1]], which is output 0 of SigmoidBackward0, is at version 2;
#  expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
#  with torch.autograd.set_detect_anomaly(True).
