import torch

torch.cuda.memory._record_memory_history()

with torch.inference_mode():
    shape = [256, 1024, 1024, 1]
    weight = torch.randn(shape, requires_grad=True, device="cuda:0")
    data = torch.randn(shape, requires_grad=False, device="cuda:0")

    x = data * weight
    mem = torch.cuda.memory_allocated()
    x.sigmoid_()
    print(f"使用原位操作产生的显存占用: {torch.cuda.memory_allocated() - mem}GB")
    mem = torch.cuda.memory_allocated()
    y = x.sigmoid()
    print(
        f"不使用原位操作产生的显存占用: {(torch.cuda.memory_allocated() - mem)/1024/1024/1024}GB"
    )

# 使用原位操作产生的显存占用: 0GB
# 不使用原位操作产生的显存占用: 1.0GB
