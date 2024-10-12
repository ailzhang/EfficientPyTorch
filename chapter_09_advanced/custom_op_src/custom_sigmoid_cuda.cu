#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input_tensor_data,
                               scalar_t* __restrict__ output_tensor_data,
                               size_t total_num_elements) {
  // Fetch thread id
  const int element_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (element_index < total_num_elements) {
    // Sigmoid Function
    scalar_t x = input_tensor_data[element_index];
    scalar_t y = 1.0 / (1.0 + exp(-x));

    // Write to output
    output_tensor_data[element_index] = y;
  }
}

torch::Tensor custom_sigmoid_cuda_forward(
    torch::Tensor input) {

  size_t total_num_elements = input.numel();

  auto output = torch::zeros_like(input);

  const int threads = 512;
  const int blocks = (total_num_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "sigmoid_kernel", ([&] {
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        total_num_elements);
  }));

  return output;
}

template <typename scalar_t>
__global__ void sigmoid_grad_kernel(const scalar_t* __restrict__ output_tensor,
                                    const scalar_t* __restrict__ output_grad_tensor,
                                    scalar_t* __restrict__ input_grad_tensor,
                                    size_t total_num_elements) {
  // Fetch thread id
  const int element_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (element_index < total_num_elements) {
    // Sigmoid Grad Function
    scalar_t output_grad = output_grad_tensor[element_index];
    scalar_t output = output_tensor[element_index];

    scalar_t input_grad = (1.0 - output) * output * output_grad;

    // Write to output
    input_grad_tensor[element_index] = input_grad;
  }
}

torch::Tensor custom_sigmoid_cuda_backward(
    torch::Tensor output,
    torch::Tensor output_grad) {

  size_t total_num_elements = output_grad.numel();

  auto input_grad = torch::zeros_like(output_grad);

  const int threads = 512;
  const int blocks = (total_num_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(output_grad.type(), "sigmoid_grad_kernel", ([&] {
    sigmoid_grad_kernel<scalar_t><<<blocks, threads>>>(
        output.data<scalar_t>(),
        output_grad.data<scalar_t>(),
        input_grad.data<scalar_t>(),
        total_num_elements);
  }));

  return input_grad;
}
