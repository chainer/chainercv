import unittest

from chainer.testing.attr import check_available
from chainer.testing.attr import gpu  # NOQA
from chainer.testing.attr import slow  # NOQA


try:
    import pytest
    pfnci_skip = pytest.mark.pfnci_skip

except ImportError:
    from chainer.testing.attr import _dummy_callable
    pfnci_skip = _dummy_callable


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
