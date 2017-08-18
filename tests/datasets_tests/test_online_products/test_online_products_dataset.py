import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import OnlineProductsDataset
from chainercv.utils import assert_is_classification_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'test'}
)
class TestOnlineProductsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = OnlineProductsDataset(split=self.split)

    @attr.slow
    def test_online_products_dataset(self):
        assert_is_classification_dataset(
            self.dataset, 22634, n_example=10)


testing.run_module(__name__, __file__)
