template <typename scalar_t>
__global__ void sigmoid_grad_kernel(
    const scalar_t *__restrict__ output_tensor,
    const scalar_t *__restrict__ output_grad_tensor,
    scalar_t *__restrict__ input_grad_tensor, size_t total_num_elements) {
    // 计算要处理的元素位置
    const int element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index < total_num_elements) {
        // 在单个元素上进行sigmoid的梯度计算
        scalar_t output_grad = output_grad_tensor[element_index];
        scalar_t output = output_tensor[element_index];
        scalar_t input_grad = (1.0 - output) * output * output_grad;
        // 将计算结果写回显存
        input_grad_tensor[element_index] = input_grad;
    }
}

torch::Tensor custom_sigmoid_cuda_backward(torch::Tensor output,
                                           torch::Tensor output_grad) {
    size_t total_num_elements = output_grad.numel();
    auto input_grad = torch::zeros_like(output_grad);
    const int threads = 512;
    const int blocks = (total_num_elements + threads - 1) / threads;

    // 将实现好的CUDA kernel注册为反向算子的CUDA后端实现
    AT_DISPATCH_FLOATING_TYPES(
        output_grad.type(), "sigmoid_grad_kernel", ([&] {
            sigmoid_grad_kernel<scalar_t><<<blocks, threads>>>(
                output.data<scalar_t>(), output_grad.data<scalar_t>(),
                input_grad.data<scalar_t>(), total_num_elements);
        }));

    return input_grad;
}
