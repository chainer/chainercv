import numpy as np
import unittest


from chainercv.utils import assert_is_point
from chainercv.utils import testing


def _random_visible_including_true(n):
    while True:
        visible = np.random.randint(0, 2, size=n).astype(np.bool)
        if visible.any():
            return visible


@testing.parameterize(
    # no visible and size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'valid': True},
    {'point': [((1., 2.), (4., 8.))],
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.int32),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 3)).astype(np.float32),
     'valid': False},
    # use visible, no size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'valid': True},
    {'point': np.random.uniform(-1, 1, size=(1, 4, 2)).astype(np.float32),
     'visible': [(True, True, True, True)],
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.int32),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10, 2)).astype(np.bool),
     'valid': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 9,)).astype(np.bool),
     'valid': False},
    # no visible, use size
    {'point': np.random.uniform(0, 32, size=(1, 10, 2)).astype(np.float32),
     'size': (32, 32),
     'valid': True},
    {'point': np.random.uniform(32, 64, size=(1, 10, 2)).astype(np.float32),
     'size': (32, 32),
     'valid': False},
    # use visible and size
    {'point': np.random.uniform(0, 32, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'size': (32, 32),
     'valid': True},
    {'point': np.random.uniform(32, 64, size=(1, 10, 2)).astype(np.float32),
     'visible': [_random_visible_including_true(10)],
     'size': (32, 32),
     'valid': False},
    # check n_point
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'n_point': 10,
     'valid': True},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'n_point': 11,
     'valid': False,
     },
    # check different instance size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(2, 10,)).astype(np.bool),
     'valid': False},
)
class TestAssertIsPoint(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, 'visible'):
            self.visible = None
        if not hasattr(self, 'size'):
            self.size = None
        if not hasattr(self, 'n_point'):
            self.n_point = None

    def test_assert_is_point(self):
        if self.valid:
            assert_is_point(
                self.point, self.visible, self.size, self.n_point)
        else:
            with self.assertRaises(AssertionError):
                assert_is_point(
                    self.point, self.visible, self.size, self.n_point)


testing.run_module(__name__, __file__)
