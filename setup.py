#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

import os


description = """
Collection of Deep Learning Computer Vision Algorithms implemented in Chainer
"""

ext_modules = [
    Extension('chainercv.utils.bbox._nms_gpu_post',
              ['chainercv/utils/bbox/_nms_gpu_post.pyx']),
]
cmdclass = {'build_ext': build_ext}

install_requires = [
    'chainer==1.24',
    'cython',
    'pillow'
]

setup(
    name='chainercv',
    version='0.4.7',
    packages=find_packages(),
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
    license='MIT',
    description=description,
    install_requires=install_requires,
    include_package_data=True,
    # for Cython
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=[np.get_include()],
)
