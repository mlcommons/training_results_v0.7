#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()


bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

batch_utils_v0p5 = CppExtension(
                        name='fairseq.data.batch_C_v0p5',
                        sources=['fairseq/data/csrc/make_batches_v0p5.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                        }
)
batch_utils_v0p5_better = CppExtension(
                        name='fairseq.data.batch_C_v0p5_better',
                        sources=['fairseq/data/csrc/make_batches_v0p5_better.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2', '--std=c++14'],
                        }
)
batch_utils_v0p6 = CppExtension(
                        name='fairseq.data.batch_C_v0p6',
                        sources=['fairseq/data/csrc/make_batches_v0p6.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2', '--std=c++14'],
                        }
)

setup(
    name='fairseq',
    version='0.5.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[bleu, batch_utils_v0p5, batch_utils_v0p5_better, batch_utils_v0p6],
    cmdclass={
                'build_ext': BuildExtension
    },
    test_suite='tests',
)
