from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for gnmt.')
with open('requirements.txt') as f:
    reqs = f.read()

extra_cuda_compile_args = {
    'cxx': ['-O2', ],
    'nvcc': ['--gpu-architecture=sm_70', ]
    }

cat_utils = CUDAExtension(
    name='seq2seq.pack_utils._C',
    sources=[
        'seq2seq/csrc/pack_utils.cpp',
        'seq2seq/csrc/pack_utils_kernel.cu'
    ],
    extra_compile_args=extra_cuda_compile_args
)

attn_score = CUDAExtension(
    name='seq2seq.attn_score._C',
    sources=[
        'seq2seq/csrc/attn_score_cuda.cpp',
        'seq2seq/csrc/attn_score_cuda_kernel.cu',
    ],
    extra_compile_args=extra_cuda_compile_args
)

setup(
    name='gnmt',
    version='0.7.0',
    description='GNMT',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[cat_utils, attn_score],
    cmdclass={
                'build_ext': BuildExtension
    },
    test_suite='tests',
)
