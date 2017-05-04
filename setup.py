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
    Extension('chainercv.links.utils.bbox.bbox_overlap',
              ['chainercv/links/utils/bbox/bbox_overlap.pyx']),
]
cmdclass = {'build_ext': build_ext}


setup(
    name='chainercv',
    version='0.4.5',
    packages=find_packages(),
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
    license='MIT',
    description=description,
    install_requires=open('requirements.txt').readlines(),
    include_package_data=True,
    data_files=[
        ('chainercv/datasets/data',
         [os.path.join('chainercv/datasets/data', fn) for fn
          in os.listdir('chainercv/datasets/data')])],
    # for Cython
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=[np.get_include()],
)
