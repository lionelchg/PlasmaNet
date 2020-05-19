from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='plasmanet_cpp',
    ext_modules=[
        CppExtension(
            'plasmanet_cpp',
            [
                'CG.cpp'
                ''
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
