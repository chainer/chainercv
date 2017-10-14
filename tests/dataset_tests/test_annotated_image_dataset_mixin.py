import unittest

import numpy as np

from chainer import testing
from chainercv.dataset import AnnotatedImageDatasetMixin


class SingleAnnotationDataset(AnnotatedImageDatasetMixin):
    def __init__(self, len):
        self._len = len

    def __len__(self):
        return self._len

    def get_image(self, i):
        return np.random.uniform(0, 256, size=(3, 32, 48)).astype(np.float32)

    def get_annotation(self, i):
        return np.random.randint(0, 9, dtype=np.int32)


class MultipleAnnotationDataset(AnnotatedImageDatasetMixin):
    def __init__(self, len):
        self._len = len

    def __len__(self):
        return self._len

    def get_image(self, i):
        return np.random.uniform(0, 256, (3, 32, 48)).astype(np.float32)

    def get_annotation(self, i):
        anno0 = np.random.uniform(0, 32, size=(20, 4)).astype(np.float32)
        anno1 = np.random.randint(0, 9, size=20, dtype=np.int32)
        return anno0, anno1


@testing.parameterize(
    {'dataset': SingleAnnotationDataset, 'n_anno': 1},
    {'dataset': MultipleAnnotationDataset, 'n_anno': 2},
)
class TestAnnotatedImageDatasetMixin(unittest.TestCase):

    def setUp(self):
        self.len = 10
        self.dataset = self.dataset(self.len)

    def test_base_dataset_len(self):
        self.assertEqual(len(self.dataset), self.len)

    def test_base_dataset_index(self):
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            self.assertIsInstance(example, tuple)
            self.assertEqual(len(example), 1 + self.n_anno)

    def test_base_dataset_slice(self):
        for example in self.dataset[:]:
            self.assertIsInstance(example, tuple)
            self.assertEqual(len(example), 1 + self.n_anno)

    def test_annotation_dataset_len(self):
        self.assertEqual(len(self.dataset.annotations), self.len)

    def test_annotation_dataset_index(self):
        for i in range(len(self.dataset.annotations)):
            example = self.dataset.annotations[i]
            if self.n_anno == 1:
                self.assertFalse(isinstance(example, tuple))
            else:
                self.assertIsInstance(example, tuple)
                self.assertEqual(len(example), self.n_anno)

    def test_annotation_dataset_slice(self):
        for example in self.dataset.annotations[:]:
            if self.n_anno == 1:
                self.assertFalse(isinstance(example, tuple))
            else:
                self.assertIsInstance(example, tuple)
                self.assertEqual(len(example), self.n_anno)


testing.run_module(__name__, __file__)
