import torch
import time
import random


def train():
    global input
    input = torch.randn(100, 100, device="cuda")


train()
print("Memory allocated for input: ", torch.cuda.memory_allocated(0))

tensors = []
for _ in range(100):
    tensors.append(torch.randn(100, 100, device="cuda"))
print("Memory allocated for tensors & input: ", torch.cuda.memory_allocated(0))

# time.sleep(1000000000000) 不管睡多久都不会释放的
# for i in range(100000000000): new_var = random.randint() 通过分配新变量触发垃圾回收，也不会清理的

print("Memory allocated total: ", torch.cuda.memory_allocated(0))


# 输出
# Memory allocated for input:  40448
# Memory allocated for tensors & input:  4085248
# Memory allocated total:  4085248
