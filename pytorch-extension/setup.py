from setuptools import setup
# this allows us to use our custom kernel in our python code
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polynomial_cuda', # name
    ext_modules=[
        CUDAExtension('polynomial_cuda', [
            'polynomial_cuda.cu', # source file
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension # build the extension
    })