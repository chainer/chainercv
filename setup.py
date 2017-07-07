#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
import pkg_resources
from setuptools import find_packages

from Cython.Distutils import build_ext as _build_ext


description = """
Collection of Deep Learning Computer Vision Algorithms implemented in Chainer
"""

ext_modules = [
    Extension('chainercv.utils.bbox._nms_gpu_post',
              ['chainercv/utils/bbox/_nms_gpu_post.pyx']),
]

setup_requires = ['numpy']
install_requires = [
    'chainer==2.0',
    'Cython',
    'Pillow'
]


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    numpy_incl not in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


cmdclass = {'build_ext': build_ext}


setup(
    name='chainercv',
    version='0.5.1',
    packages=find_packages(),
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
    license='MIT',
    description=description,
    setup_requires=setup_requires,
    install_requires=install_requires,
    include_package_data=True,
    # for Cython
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
