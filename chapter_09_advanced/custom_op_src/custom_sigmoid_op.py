import torch
from torch.autograd import Function
import custom_ops


class CustomSigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = custom_ops.sigmoid_fwd(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = custom_ops.sigmoid_bwd(output, grad_output.contiguous())
        return grad_input


class CustomSigmoid(torch.nn.Module):
    def forward(self, input):
        return CustomSigmoidFunction.apply(input)
