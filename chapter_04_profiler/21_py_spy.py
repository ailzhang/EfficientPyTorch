import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def numpy_heavy_computation(input_array):
    size_inner = 1000
    size_0 = input_array.shape[0]
    size_1 = input_array.shape[1]
    result = input_array
    for _ in range(2):
        matrix_a = np.random.randn(size_0, size_inner)
        matrix_b = np.random.randn(size_inner, size_1)
        result = np.dot(matrix_a, matrix_b) + result
    return result


def run(data, model):
    processed_data = numpy_heavy_computation(data)
    tensor_data = torch.tensor(
        processed_data[:10, :10], dtype=torch.float32, device="cuda"
    )
    output = model(tensor_data)


def main():
    model = SimpleModel().to("cuda")
    data = np.random.randn(10, 10)
    for i in range(1000):
        run(data, model)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
