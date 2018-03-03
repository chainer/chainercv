import unittest

import numpy as np

from chainer import testing

from chainer.datasets import TupleDataset
from chainercv.datasets import MixUpSoftLabelDataset
from chainercv.datasets import SiameseDataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


N = 15


@testing.parameterize(
    # Positive and negative samples
    {'labels_0': np.arange(N, dtype=np.int32) % 3,
     'labels_1': np.arange(N, dtype=np.int32) % 3,
     'pos_exist': True, 'neg_exist': True,
     },
    # No positive
    {'labels_0': np.zeros(N, dtype=np.int32),
     'labels_1': np.ones(N, dtype=np.int32),
     'pos_exist': False, 'neg_exist': True
     },
    # No negative
    {'labels_0': np.ones(N, dtype=np.int32),
     'labels_1': np.ones(N, dtype=np.int32),
     'pos_exist': True, 'neg_exist': False},
)
class TestMixupDataset(unittest.TestCase):

    img_shape = (3, 32, 48)

    def setUp(self):
        np.random.shuffle(self.labels_0)
        np.random.shuffle(self.labels_1)

        dataset_0 = TupleDataset(
            np.random.uniform(size=(N,) + self.img_shape), self.labels_0)
        dataset_1 = TupleDataset(
            np.random.uniform(size=(N,) + self.img_shape), self.labels_1)
        self.n_class = np.max((self.labels_0, self.labels_1)) + 1
        self.siamese_dataset = SiameseDataset(dataset_0, dataset_1)

    def _check_example(self, example):
        assert_is_image(example[0])
        self.assertEqual(example[0].shape, self.img_shape)
        self.assertEqual(example[1].dtype, np.float32)
        self.assertEqual(example[1].ndim, 1)
        self.assertEqual(example[1].shape[0], self.n_class + 1)
        self.assertAlmostEqual(example[1].sum(), 1.0)

    def test_mixup(self):
        dataset = MixUpSoftLabelDataset(self.siamese_dataset, self.n_class)
        for i in range(10):
            example = dataset[i]
            self._check_example(example)
        self.assertEqual(len(dataset), N)


testing.run_module(__name__, __file__)
