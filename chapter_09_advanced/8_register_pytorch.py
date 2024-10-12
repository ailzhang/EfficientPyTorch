import torch
from torch.autograd import Function

# custom_ops 便是我们自定义的Python扩展模块，包含了C++中编写的自定义sigmoid算子
import custom_ops


class CustomSigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # 调用自定义算子的前向操作
        output = custom_ops.sigmoid_fwd(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        # 调用自定义算子的反向操作
        grad_input = custom_ops.sigmoid_bwd(output, grad_output.contiguous())
        return grad_input


class CustomSigmoid(torch.nn.Module):
    def forward(self, input):
        return CustomSigmoidFunction.apply(input)
