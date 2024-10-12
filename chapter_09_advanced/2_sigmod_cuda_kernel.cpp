#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t *__restrict__ input_tensor_data,
                               scalar_t *__restrict__ output_tensor_data,
                               size_t total_num_elements) {
    // 计算要处理的元素位置
    const int element_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_index < total_num_elements) {
        // 在单个元素上进行sigmoid计算
        scalar_t x = input_tensor_data[element_index];
        scalar_t y = 1.0 / (1.0 + exp(-x));

        // 将计算结果写回显存
        output_tensor_data[element_index] = y;
    }
}
