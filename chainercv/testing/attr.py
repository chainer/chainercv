from chainer.testing.attr import cudnn  # NOQA
from chainer.testing.attr import gpu  # NOQA
from chainer.testing.attr import multi_gpu  # NOQA
from chainer.testing.attr import slow  # NOQA

try:
    import pytest
    disk = pytest.mark.disk
except ImportError:
    from chainer.testing.attr import _dummy_callable
    disk = _dummy_callable
