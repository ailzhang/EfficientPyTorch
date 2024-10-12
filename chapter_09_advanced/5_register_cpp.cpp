#include <torch/extension.h>

#include <iostream>
#include <vector>

// forward declarations or include the header
torch::Tensor custom_sigmoid_cuda_forward(torch::Tensor input);

torch::Tensor custom_sigmoid_cuda_backward(torch::Tensor output,
                                           torch::Tensor output_grad);

// 简易的sigmoid前向算子的CPU后端实现
torch::Tensor custom_sigmoid_cpu_forward(torch::Tensor input) {
    return 1.0 / (1 + torch::exp(-input));
}

// 简易的sigmoid反向算子的CPU后端实现
torch::Tensor custom_sigmoid_cpu_backward(torch::Tensor output,
                                          torch::Tensor output_grad) {
    return (1 - output) * output * output_grad;
}

// 进行前向算子的后端实现分发
torch::Tensor custom_sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous")

    if (input.device().is_cuda()) {
        return custom_sigmoid_cuda_forward(input);
    } else {
        return custom_sigmoid_cpu_forward(input);
    }
}

// 进行反向算子的后端实现分发
torch::Tensor custom_sigmoid_backward(torch::Tensor output,
                                      torch::Tensor grad_output) {
    TORCH_CHECK(grad_output.is_contiguous(), "input must be contiguous")

    if (output.device().is_cuda()) {
        return custom_sigmoid_cuda_backward(output, grad_output);
    } else {
        return custom_sigmoid_cpu_backward(output, grad_output);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 注册算子以便在Python中调用
    m.def("sigmoid_fwd", &custom_sigmoid_forward, "Custom sigmoid forward");
    m.def("sigmoid_bwd", &custom_sigmoid_backward, "Custom sigmoid backward");
}
