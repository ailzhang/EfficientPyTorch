import torch
import torch.optim as optim


torch.cuda.memory._record_memory_history()

shape = [256, 1024, 1024, 1]
weight = torch.randn(shape, requires_grad=True, device="cuda:0")
data = torch.randn(shape, requires_grad=False, device="cuda:0")

x = data * weight
x = x * weight
x = x.sum()

torch.cuda.memory._dump_snapshot("triple_muls_fwd.pickle")

optimizer = optim.SGD([weight], lr=0.01)
optimizer.zero_grad()

x.backward()

optimizer.step()

torch.cuda.memory._dump_snapshot("traces/double_muls_full.pickle")
