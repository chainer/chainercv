import numpy as np
import unittest

from chainer import testing

from chainercv.utils import assert_is_point


def _random_visible_including_true(n):
    while True:
        visible = np.random.randint(0, 2, size=n).astype(np.bool)
        if visible.any():
            return visible


@testing.parameterize(
    # no visible and size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'no_error': True},
    {'point': [((1., 2.), (4., 8.))],
     'no_error': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.int32),
     'no_error': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 3)).astype(np.float32),
     'no_error': False},
    # use visible, no size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'no_error': True},
    {'point': np.random.uniform(-1, 1, size=(1, 4, 2)).astype(np.float32),
     'visible': [(True, True, True, True)],
     'no_error': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.int32),
     'no_error': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10, 2)).astype(np.bool),
     'no_error': False},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 9,)).astype(np.bool),
     'no_error': False},
    # no visible, use size
    {'point': np.random.uniform(0, 32, size=(1, 10, 2)).astype(np.float32),
     'size': (32, 32),
     'no_error': True},
    {'point': np.random.uniform(32, 64, size=(1, 10, 2)).astype(np.float32),
     'size': (32, 32),
     'no_error': False},
    # use visible and size
    {'point': np.random.uniform(0, 32, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'size': (32, 32),
     'no_error': True},
    {'point': np.random.uniform(32, 64, size=(1, 10, 2)).astype(np.float32),
     'visible': [_random_visible_including_true(10)],
     'size': (32, 32),
     'no_error': False},
    # check n_point
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'n_point': 10,
     'no_error': True},
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(1, 10,)).astype(np.bool),
     'n_point': 11,
     'no_error': False,
     },
    # check different instance size
    {'point': np.random.uniform(-1, 1, size=(1, 10, 2)).astype(np.float32),
     'visible': np.random.randint(0, 2, size=(2, 10,)).astype(np.bool),
     'no_error': False},
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
        if self.no_error:
            assert_is_point(
                self.point, self.visible, self.size, self.n_point)
        else:
            with self.assertRaises(AssertionError):
                assert_is_point(
                    self.point, self.visible, self.size, self.n_point)


testing.run_module(__name__, __file__)
