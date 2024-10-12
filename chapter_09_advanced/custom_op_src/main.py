import torch
import torch.nn.functional as F
import numpy as np
from custom_sigmoid_op import CustomSigmoid


def run(np_input, sigmoid_op, device="cuda"):
    x = torch.tensor(np_input, dtype=torch.double, device=device, requires_grad=True)
    output = sigmoid_op(x)

    loss = torch.sum(output)
    loss.backward()

    return output.clone(), x.grad.clone()


custom_sigmoid = CustomSigmoid()

device = "cuda"

# Prepare a random input tensor
np_input = np.random.randn(10, 20)

for device in ["cpu", "cuda"]:
    sigmoid_out_torch, sigmoid_grad_torch = run(np_input, torch.sigmoid, device)
    sigmoid_out_custom, sigmoid_grad_custom = run(np_input, custom_sigmoid, device)

    # Compare results
    if torch.allclose(sigmoid_out_torch, sigmoid_out_custom) and torch.allclose(
        sigmoid_grad_torch, sigmoid_grad_custom
    ):
        print(f"Pass on {device}")
    else:
        print(f"Error: results mismatch on {device}")
