import torch

original = torch.rand((2, 12))

reshaped = original.view(2, 3, 4)
print("reshaped shape:", reshaped.shape)
# reshaped shape: torch.Size([2, 3, 4])


flattened = reshaped.view(-1)
print("flattened shape:", flattened.shape)
# flattened shape: torch.Size([24])
