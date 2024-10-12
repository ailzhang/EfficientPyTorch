import torch


def func():
    tensors = []
    for _ in range(100):
        tensors.append(torch.randn(100, 100, device="cuda"))

    print("Memory allocated from function: ", torch.cuda.memory_allocated(0))
    return


func()
print("Memory allocated: ", torch.cuda.memory_allocated(0))

# 输出:
# Memory allocated from function:  4044800
# Memory allocated:  0
