import torch


class MyMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return input1 * input1 * input2

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = grad_output * 2 * input1 * input2
        grad_input2 = grad_output * input1 * input1
        return grad_input1, grad_input2


# 使用自定义的乘法操作
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)
z = MyMul.apply(x, y)
z.backward(torch.tensor([1.0, 1.0]))

print(f"x.grad={x.grad}, y.grad={y.grad}")
# x.grad=tensor([12., 24.]), y.grad=tensor([4., 9.])
