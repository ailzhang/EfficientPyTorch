import torch

t1 = torch.randn([1024, 1024], device="cuda:0")  # 4MB

shape = [256, 1024, 1024, 1]  # 1024MB
t2 = torch.randn(shape, device="cuda:0")

print(
    f"PyTorch reserved {torch.cuda.memory_reserved()/1024/1024}MB, allocated {torch.cuda.memory_allocated()/1024/1024}MB"
)
# PyTorch reserved 1044.0MB, allocated 1028.0MB
