import numpy as np
import unittest

from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.utils import apply_prediction_to_iterator
from chainercv.utils import ProgressHook


class TestProgressHook(unittest.TestCase):

    def setUp(self):
        def predict(imgs):
            n_img = len(imgs)
            return [np.random.uniform() for _ in range(n_img)]

        self.predict = predict

        self.dataset = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            self.dataset.append(np.random.randint(0, 256, size=(3, H, W)))

    def test_progress_hook(self):
        iterator = SerialIterator(self.dataset, 2, repeat=False)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            self.predict, iterator,
            hook=ProgressHook(n_total=len(self.dataset)))

        for _ in imgs:
            pass

    def test_progress_hook_with_infinite_iterator(self):
        iterator = SerialIterator(self.dataset, 2)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            self.predict, iterator)

        for _ in range(10):
            next(imgs)


testing.run_module(__name__, __file__)
