from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="rgb_lut_cuda",
    packages=["rgb_lut_cuda"],
    ext_modules=[
        CUDAExtension(
            name="_rgb_lut_cuda_ext",
            sources=["csrc/rgb_lut.cpp", "csrc/rgb_lut_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", '-arch=sm_86']},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
