import numpy as np
import unittest

from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.utils import apply_prediction_to_iterator
from chainercv.utils import ProgressHook


@testing.parameterize(*testing.product({
    'multi_pred_values': [False, True],
    'with_gt_values': [False, True],
}))
class TestProgressHook(unittest.TestCase):

    def test_progress_hook(self):
        def predict(imgs):
            n_img = len(imgs)
            return [np.random.uniform() for _ in range(n_img)]

        dataset = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            dataset.append(np.random.randint(0, 256, size=(3, H, W)))
        iterator = SerialIterator(dataset, 2, repeat=False)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            predict, iterator, hook=ProgressHook(n_total=len(dataset)))

        for _ in imgs:
            pass


class TestProgressHookWithInfiniteIterator(unittest.TestCase):

    def test_progress_hook(self):
        def predict(imgs):
            n_img = len(imgs)
            return [np.random.uniform() for _ in range(n_img)]

        dataset = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            dataset.append(np.random.randint(0, 256, size=(3, H, W)))
        iterator = SerialIterator(dataset, 2)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            predict, iterator)

        for _ in range(10):
            next(imgs)


testing.run_module(__name__, __file__)
