from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy as np


ext_modules = [
    Extension('lib.bbox', ['lib/cython/bbox.pyx']),
    Extension('lib.nms_cpu', ['lib/cython/nms_cpu.pyx'])
]
cmdclass = {'build_ext': build_ext}
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=[np.get_include()]
)
