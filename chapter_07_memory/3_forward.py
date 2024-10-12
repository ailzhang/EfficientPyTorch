import torch

torch.cuda.memory._record_memory_history()

with torch.inference_mode():
    shape = [256, 1024, 1024, 1]
    weight = torch.randn(shape, device="cuda:0")  # (1)
    data = torch.randn(shape, device="cuda:0")  # (2)

    x = data * weight  # (3)
    x = x * weight  # (4)
    x = x.sum()

torch.cuda.memory._dump_snapshot("traces/double_muls_inference.pickle")
