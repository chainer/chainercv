import unittest

import numpy as np

from chainer import testing

from chainer.datasets import TupleDataset
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
class TestSiameseDataset(unittest.TestCase):

    img_shape = (3, 32, 48)

    def setUp(self):
        np.random.shuffle(self.labels_0)
        np.random.shuffle(self.labels_1)

        self.dataset_0 = TupleDataset(
            np.random.uniform(size=(N,) + self.img_shape), self.labels_0)
        self.dataset_1 = TupleDataset(
            np.random.uniform(size=(N,) + self.img_shape), self.labels_1)
        self.n_class = np.max((self.labels_0, self.labels_1)) + 1

    def _check_example(self, example):
        assert_is_image(example[0])
        self.assertEqual(example[0].shape, self.img_shape)
        assert_is_image(example[2])
        self.assertEqual(example[2].shape, self.img_shape)

        self.assertIsInstance(example[1], np.int32)
        self.assertEqual(example[1].ndim, 0)
        self.assertTrue(example[1] >= 0 and example[1] < self.n_class)
        self.assertIsInstance(example[3], np.int32)
        self.assertEqual(example[3].ndim, 0)
        self.assertTrue(example[3] >= 0 and example[1] < self.n_class)

    def test_no_pos_ratio(self):
        dataset = SiameseDataset(self.dataset_0, self.dataset_1)
        for i in range(10):
            example = dataset[i]
            self._check_example(example)
        self.assertEqual(len(dataset), N)

    def test_pos_ratio(self):
        if self.pos_exist and self.neg_exist:
            dataset = SiameseDataset(self.dataset_0, self.dataset_1, 0.5,
                                     labels_0=self.labels_0,
                                     labels_1=self.labels_1)
            for i in range(10):
                example = dataset[i]
                self._check_example(example)
            self.assertEqual(len(dataset), N)
        else:
            with self.assertRaises(ValueError):
                dataset = SiameseDataset(self.dataset_0, self.dataset_1, 0.5,
                                         labels_0=self.labels_0,
                                         labels_1=self.labels_1)

    def test_pos_ratio_equals_0(self):
        if self.neg_exist:
            dataset = SiameseDataset(self.dataset_0, self.dataset_1, 0)

            for i in range(10):
                example = dataset[i]
                self._check_example(example)
                if self.neg_exist:
                    self.assertNotEqual(example[1], example[3])
            self.assertEqual(len(dataset), N)
        else:
            with self.assertRaises(ValueError):
                dataset = SiameseDataset(self.dataset_0, self.dataset_1, 0)

    def test_pos_ratio_equals_1(self):
        if self.pos_exist:
            dataset = SiameseDataset(self.dataset_0, self.dataset_1, 1)

            for i in range(10):
                example = dataset[i]
                self._check_example(example)
                if self.pos_exist:
                    self.assertEqual(example[1], example[3])
            self.assertEqual(len(dataset), N)
        else:
            with self.assertRaises(ValueError):
                dataset = SiameseDataset(self.dataset_0, self.dataset_1, 1)

    def test_length_manual(self):
        dataset = SiameseDataset(self.dataset_0, self.dataset_1, length=100)
        self.assertEqual(len(dataset), 100)


testing.run_module(__name__, __file__)
