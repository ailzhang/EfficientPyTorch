torch::Tensor custom_sigmoid_cuda_forward(torch::Tensor input) {
    size_t total_num_elements = input.numel();

    auto output = torch::zeros_like(input);

    const int threads = 512;
    const int blocks = (total_num_elements + threads - 1) / threads;

    // 将实现好的CUDA kernel注册为前向算子的CUDA后端实现
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "sigmoid_kernel", ([&] {
            sigmoid_kernel<scalar_t><<<blocks, threads>>>(
                input.data<scalar_t>(), output.data<scalar_t>(),
                total_num_elements);
        }));

    return output;
}
