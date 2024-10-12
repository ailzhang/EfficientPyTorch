import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

model = nn.Sequential(
    nn.Linear(1000, 40000),
    nn.ReLU(),
    nn.Linear(40000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 5),
    nn.ReLU(),
).to("cuda")

input_var = torch.randn(10, 1000, device="cuda", requires_grad=True)

segments = 2
modules = [module for k, module in model._modules.items()]

# (1). 使用checkpoint技术
out = checkpoint_sequential(modules, segments, input_var)

model.zero_grad()
out.sum().backward()
print(f"使用checkpoint技术显存分配峰值: {torch.cuda.max_memory_allocated()/1024/1024}MB")
# 使用checkpoint技术显存分配峰值: 628.63671875MB

out_checkpointed = out.data.clone()
grad_checkpointed = {}
for name, param in model.named_parameters():
    grad_checkpointed[name] = param.grad.data.clone()

# (2). 不使用checkpoint技术
original = model
x = input_var.clone().detach_()
out = original(x)

out_not_checkpointed = out.data.clone()

original.zero_grad()
out.sum().backward()
print(f"不使用checkpoint技术显存分配峰值: {torch.cuda.max_memory_allocated()/1024/1024}MB")
# 不使用checkpoint技术显存分配峰值: 936.17431640625MB

grad_not_checkpointed = {}
for name, param in model.named_parameters():
    grad_not_checkpointed[name] = param.grad.data.clone()


# 对比使用和不使用checkpoint技术计算出来的梯度都是一样的
assert torch.allclose(out_checkpointed, out_not_checkpointed)
for name in grad_checkpointed:
    assert torch.allclose(grad_checkpointed[name], grad_not_checkpointed[name])
