import numpy as np
import unittest

from chainer import testing

from chainercv.utils import assert_is_bbox


@testing.parameterize(
    {
        'tl': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
        'hw': np.random.uniform(0.1, 1, size=(10, 2)).astype(np.float32),
        'valid': True},
    {
        'tl': np.random.uniform(0, 32, size=(10, 2)).astype(np.float32),
        'hw': np.random.uniform(0.1, 16, size=(10, 2)).astype(np.float32),
        'size': (48, 64),
        'valid': True},
    {
        'tl': np.random.uniform(20, 32, size=(10, 2)).astype(np.float32),
        'hw': np.random.uniform(16, 32, size=(10, 2)).astype(np.float32),
        'size': (16, 24),
        'valid': False},
    {
        'tl': np.random.uniform(-1, 1, size=(10, 2)).astype(float),
        'hw': np.random.uniform(0.1, 1, size=(10, 2)).astype(float),
        'valid': False},
    {
        'tl': np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32),
        'hw': np.random.uniform(-1, 0, size=(10, 2)).astype(np.float32),
        'valid': False},
    {
        'bbox': np.random.uniform(-1, 1, size=(10, 5)).astype(np.float32),
        'valid': False},
    {
        'bbox': np.random.uniform(-1, 1, size=(10, 4, 1)).astype(np.float32),
        'valid': False},
    {
        'bbox': ((0, 1, 2, 3), (1, 2, 3, 4)),
        'valid': False},
)
class TestAssertIsBbox(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, 'bbox'):
            self.bbox = np.hstack((self.tl, self.tl + self.hw))
        if not hasattr(self, 'size'):
            self.size = None

    def test_assert_is_bbox(self):
        if self.valid:
            assert_is_bbox(self.bbox, self.size)
        else:
            with self.assertRaises(AssertionError):
                assert_is_bbox(self.bbox, self.size)


testing.run_module(__name__, __file__)
