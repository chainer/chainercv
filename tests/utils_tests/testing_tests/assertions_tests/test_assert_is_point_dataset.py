import numpy as np
import unittest

from chainer.dataset import DatasetMixin
from chainer import testing

from chainercv.utils import assert_is_point_dataset


class PointDataset(DatasetMixin):

    H = 48
    W = 64

    def __init__(self, n_point_candidates,
                 return_visible, *options):
        self.n_point_candidates = n_point_candidates
        self.return_visible = return_visible
        self.options = options

    def __len__(self):
        return 10

    def get_example(self, i):
        n_inst = 2
        img = np.random.randint(0, 256, size=(3, self.H, self.W))
        n_point = np.random.choice(self.n_point_candidates)
        point_y = np.random.uniform(0, self.H, size=(n_inst, n_point))
        point_x = np.random.uniform(0, self.W, size=(n_inst, n_point))
        point = np.stack((point_y, point_x), axis=2).astype(np.float32)
        if self.return_visible:
            visible = np.random.randint(
                0, 2, size=(n_inst, n_point)).astype(np.bool)
            return (img, point, visible) + self.options
        else:
            return (img, point) + self.options


class InvalidSampleSizeDataset(PointDataset):

    def get_example(self, i):
        img = super(
            InvalidSampleSizeDataset, self).get_example(i)[0]
        return img


class InvalidImageDataset(PointDataset):

    def get_example(self, i):
        img = super(
            InvalidImageDataset, self).get_example(i)[0]
        rest = super(
            InvalidImageDataset, self).get_example(i)[1:]
        return (img[0],) + rest


class InvalidPointDataset(PointDataset):

    def get_example(self, i):
        img, point = super(InvalidPointDataset, self).get_example(i)[:2]
        rest = super(InvalidPointDataset, self).get_example(i)[2:]
        point += 1000
        return (img, point) + rest


@testing.parameterize(
    # No optional Values
    {'dataset': PointDataset([10, 15], True), 'valid': True, 'n_point': None},
    {'dataset': PointDataset([10, 15], False), 'valid': True, 'n_point': None},
    {'dataset': PointDataset([15], True), 'valid': True, 'n_point': 15},
    {'dataset': PointDataset([15], False), 'valid': True, 'n_point': 15},
    # Invalid n_point
    {'dataset': PointDataset([15], True), 'valid': False, 'n_point': 10},
    {'dataset': PointDataset([15], False), 'valid': False, 'n_point': 10},
    # Return optional values
    {'dataset': PointDataset([10, 15], True, 'option'),
     'valid': True, 'n_point': None},
    {'dataset': PointDataset([10, 15], False, 'option'),
     'valid': True, 'n_point': None, 'no_visible': True},
    {'dataset': PointDataset([15], True, 'option'),
     'valid': True, 'n_point': 15},
    {'dataset': PointDataset([15], False, 'option'),
     'valid': True, 'n_point': 15, 'no_visible': True},
    # Invalid datasets
    {'dataset': InvalidSampleSizeDataset([10], True),
     'valid': False, 'n_point': None},
    {'dataset': InvalidImageDataset([10], True),
     'valid': False, 'n_point': None},
    {'dataset': InvalidPointDataset([10], True),
     'valid': False, 'n_point': None},
)
class TestAssertIsPointDataset(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, 'no_visible'):
            self.no_visible = False

    def test_assert_is_point_dataset(self):
        if self.valid:
            assert_is_point_dataset(
                self.dataset, self.n_point, 20, self.no_visible)
        else:
            with self.assertRaises(AssertionError):
                assert_is_point_dataset(
                    self.dataset, self.n_point, 20, self.no_visible)


testing.run_module(__name__, __file__)
