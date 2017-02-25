from distutils.core import setup

from Cython.Build import cythonize

import numpy as np

setup(
    ext_modules=cythonize(['lib/cython/bbox.pyx',
                           'lib/cython/nms_cpu.pyx']),
    include_dirs=[np.get_include()]
)
