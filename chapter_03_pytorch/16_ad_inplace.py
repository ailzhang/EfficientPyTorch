import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

t = x * 10
z = t * t

# 原位加法破坏了反向计算图需要的中间结果
t.add_(1)
# 触发报错
#     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3]], which is output 0 of AddBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).


loss = z.mean()

loss.backward()

print(x.grad)
