#!/usr/bin/env python

"""
Parts of this file were taken from Pandas project,
(https://github.com/pandas-dev/pandas)
which have released under BSD 3-clause "New" or "Revised" License.

"""

from distutils.core import setup
import os
import pkg_resources
from setuptools.command import sdist
from setuptools import find_packages

from distutils.extension import Extension


description = """
Collection of Deep Learning Computer Vision Algorithms implemented in Chainer
"""


setup_requires = ['numpy']
install_requires = [
    'chainer>=6.0',
    'Pillow'
]

ext_data = {
    'utils.bbox._nms_gpu_post': {'pyxfile': 'utils/bbox/_nms_gpu_post'}
}

try:
    from Cython.Distutils import build_ext as _build_ext
    use_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext
    use_cython = False

for name, data in ext_data.items():
    src = os.path.join('chainercv', data['pyxfile'] + '.pyx')
    if not os.path.exists(src):
        use_cython = False
        break

suffix = '.pyx' if use_cython else '.c'


extensions = []
for name, data in ext_data.items():
    sources = [os.path.join('chainercv', data['pyxfile'] + suffix)]

    extensions.append(
        Extension('chainercv.{}'.format(name),
                  sources=sources)
    )


class CheckingBuildExt(_build_ext):

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    print("{}: -> [{}]".format(ext.name, ext.sources))
                    raise Exception("""Cython-generated file '{}' not found.
                Cython is required to compile chainercv from a development
                branch. Please install Cython or
                download a release package of chainercv.
                """.format(src))

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)

        # Include NumPy
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')
        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    numpy_incl not in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


class Sdist(sdist.sdist):

    def __init__(self, *args, **kwargs):
        assert use_cython

        from Cython.Build import cythonize

        for e in extensions:
            for src in e.sources:
                cythonize(src)

        super(sdist.sdist, self).__init__(*args, **kwargs)


cmdclass = {
    'build_ext': CheckingBuildExt,
    'sdist': Sdist,
}


setup(
    name='chainercv',
    version='0.12.0',
    packages=find_packages(),
    author='Yusuke Niitani, Toru Ogawa',
    author_email='niitani@preferred.jp, ogawa@preferred.jp',
    license='MIT',
    description=description,
    setup_requires=setup_requires,
    install_requires=install_requires,
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
)
