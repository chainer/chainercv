import numpy as np
import unittest

from chainer.dataset import DatasetMixin
from chainer import testing

from chainercv.utils import assert_is_label_dataset


class LabelDataset(DatasetMixin):

    def __init__(self, color, *options):
        self.color = color
        self.options = options

    def __len__(self):
        return 10

    def get_example(self, i):
        if self.color:
            img = np.random.randint(0, 256, size=(3, 48, 64))
        else:
            img = np.random.randint(0, 256, size=(1, 48, 64))
        label = np.random.randint(0, 20, dtype='i')
        return (img, label) + self.options


class InvalidSampleSizeDataset(LabelDataset):

    def get_example(self, i):
        img, label = super(
            InvalidSampleSizeDataset, self).get_example(i)[:2]
        return img


class InvalidImageDataset(LabelDataset):

    def get_example(self, i):
        img, label = super(InvalidImageDataset, self).get_example(i)[:2]
        return img[0], label


class InvalidLabelDataset(LabelDataset):

    def get_example(self, i):
        img, label = super(InvalidLabelDataset, self).get_example(i)[:2]
        label += 1000
        return img, label


@testing.parameterize(*(
    testing.product_dict(
        [
            {'dataset': LabelDataset, 'valid': True},
            {'dataset': LabelDataset, 'valid': True,
             'option': 'option'},
            {'dataset': InvalidSampleSizeDataset, 'valid': False},
            {'dataset': InvalidImageDataset, 'valid': False},
            {'dataset': InvalidLabelDataset, 'valid': False}
        ],
        [
            {'color': False},
            {'color': True}
        ]
    )
))
class TestAssertIsLabelDataset(unittest.TestCase):

    def test_assert_is_label_dataset(self):
        if hasattr(self, 'option'):
            dataset = self.dataset(self.color, self.option)
        else:
            dataset = self.dataset(self.color)

        if self.valid:
            assert_is_label_dataset(dataset, 20, color=self.color)
        else:
            with self.assertRaises(AssertionError):
                assert_is_label_dataset(dataset, 20, color=self.color)


testing.run_module(__name__, __file__)
