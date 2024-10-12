from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="custom_ops",
    ext_modules=[
        CppExtension(
            "custom_ops",
            [
                "custom_sigmoid.cpp",
                "custom_sigmoid_cuda.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
