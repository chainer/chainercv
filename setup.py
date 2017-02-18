#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages


setup(
    name='chainer_cv',
    version='0.0.1',
    packages=find_packages(),
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
    license='MIT',
    description='Collection of Deep Learning Computer Vision Algorithms implemented in Chainer',
    install_requires=open('requirements.txt').readlines(),
)
