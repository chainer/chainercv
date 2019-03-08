import unittest

from chainer.testing.attr import check_available
from chainer.testing.attr import gpu  # NOQA
from chainer.testing.attr import slow  # NOQA


def mpi(f):
    check_available()
    import pytest

    try:
        import mpi4py.MPI  # NOQA
        available = True
    except ImportError:
        available = False

    return unittest.skipUnless(
        available, 'mpi4py is not installed')(pytest.mark.mpi(f))
