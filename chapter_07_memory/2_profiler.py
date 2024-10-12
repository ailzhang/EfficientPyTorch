import torch

torch.cuda.memory._record_memory_history()


with torch.inference_mode():
    shape = [256, 1024, 1024, 1]
    x1 = torch.randn(shape, device="cuda:0")
    x2 = torch.randn(shape, device="cuda:0")

    # Multiplication
    y = x1 * x2

torch.cuda.memory._dump_snapshot("traces/vram_profile_example.pickle")
