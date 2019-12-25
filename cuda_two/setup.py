from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda_two',
    ext_modules=[
        CUDAExtension('lltm_cuda_two', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
