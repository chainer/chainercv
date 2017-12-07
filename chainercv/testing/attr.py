from chainer.testing import attr

from attr import cudnn  # NOQA
from attr import slow  # NOQA
from attr import multi_gpu  # NOQA
from attr import gpu  # NOQA

try:
    import pytest
    disk = pytest.mark.disk
except ImportError:
    disk = attr._dummy_callable
