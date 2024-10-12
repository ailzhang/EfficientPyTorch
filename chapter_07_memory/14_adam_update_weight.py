import torch
import torch.optim as optim


# 模拟模型参数
def generate_params(device, shape):
    params = [
        torch.rand(shape, dtype=torch.float32, requires_grad=True, device=device)
        for _ in range(6)
    ]
    return params


# 模拟模型运行
def run(params):
    x = torch.rand(shape, dtype=torch.float32, device=device)
    x = params[0] * x
    x = params[1] * x
    x = params[2] * x
    x = params[3] * x
    x = params[4] * x
    x = params[5] * x
    x = x.sum()
    return x


# (1) 使用for-each进行参数更新
torch.cuda.memory._record_memory_history()
device = "cuda:0"
shape = [4]
params = generate_params(device, shape)
out = run(params)

optimizer = optim.Adam(params, lr=0.01, foreach=True)
optimizer.zero_grad()

out.backward()
optimizer.step()

torch.cuda.memory._dump_snapshot("traces/adam_foreach.pickle")

# (2) 使用for-loop进行参数更新
torch.cuda.memory._record_memory_history()

device = "cuda:0"
shape = [4]
params = generate_params(device, shape)
out = run(params)

optimizer = optim.Adam(params, lr=0.01, foreach=False)
optimizer.zero_grad()

out.backward()
optimizer.step()

torch.cuda.memory._dump_snapshot("traces/adam_forloop.pickle")
