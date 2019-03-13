import numpy as np
import unittest


from chainercv.utils import assert_is_point
from chainercv.utils import testing


def _random_mask_including_true(n):
    while True:
        mask = np.random.randint(0, 2, size=n).astype(np.bool)
        if mask.any():
            return mask


@testing.parameterize(
    # no mask and size
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
     'valid': True},
    {'point': ((1., 2.), (4., 8.)),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.int32),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(10, 3)).astype(np.float32),
     'valid': False},
    # use mask, no size
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
     'mask': np.random.randint(0, 2, size=(10,)).astype(np.bool),
     'valid': True},
    {'point': np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32),
     'mask': (True, True, True, True),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
     'mask': np.random.randint(0, 2, size=(10,)).astype(np.int32),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
     'mask': np.random.randint(0, 2, size=(10, 2)).astype(np.bool),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
     'mask': np.random.randint(0, 2, size=(9,)).astype(np.bool),
     'valid': False},
    # no mask, use size
    {'point': np.random.uniform(0, 32, size=(10, 2)).astype(np.float32),
     'size': (32, 32),
     'valid': True},
    {'point': np.random.uniform(32, 64, size=(10, 2)).astype(np.float32),
     'size': (32, 32),
     'valid': False},
    # use mask and size
    {'point': np.random.uniform(0, 32, size=(10, 2)).astype(np.float32),
     'mask': np.random.randint(0, 2, size=(10,)).astype(np.bool),
     'size': (32, 32),
     'valid': True},
    {'point': np.random.uniform(32, 64, size=(10, 2)).astype(np.float32),
     'mask': _random_mask_including_true(10),
     'size': (32, 32),
     'valid': False},
)
class TestAssertIsPoint(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, 'mask'):
            self.mask = None
        if not hasattr(self, 'size'):
            self.size = None

    def test_assert_is_point(self):
        if self.valid:
            assert_is_point(self.point, self.mask, self.size)
        else:
            with self.assertRaises(AssertionError):
                assert_is_point(self.point, self.mask, self.size)


testing.run_module(__name__, __file__)
