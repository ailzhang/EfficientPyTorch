import torch
import torch.optim as optim

torch.manual_seed(1000)

N = 128
Total_N = 512
dataset = torch.randn([Total_N, 32, 1024], requires_grad=False)

weight = torch.randn([1024, 32], requires_grad=True, device="cuda:0")
optimizer = optim.SGD([weight], lr=0.01)

num_iters = int(Total_N / 256)
steps = 2

for i in range(num_iters):
    # 模拟一个批次的训练
    optimizer.zero_grad()

    for j in range(steps):
        offset = i * 256 + N * j

        input = dataset[offset : offset + N, :, :].to(torch.device("cuda:0"))
        y = input.matmul(weight)
        loss = y.sum()

        loss.backward()
    optimizer.step()

print(weight.sum())
print(f"显存分配的峰值: {torch.cuda.max_memory_allocated()/1024/1024}MB")

# 输出：
# tensor(2096.2283, device='cuda:0', grad_fn=<SumBackward0>)
# 显存分配的峰值: 49.00048828125MB
