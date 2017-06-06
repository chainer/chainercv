import numpy as np
from six.moves import zip_longest
import unittest

import chainer
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.utils import apply_prediction_to_iterator


@testing.parameterize(*testing.product({
    'multi_pred_values': [False, True],
    'with_gt_values': [False, True],
    'with_hook': [False, True],
}))
class TestApplyPredictionToIterator(unittest.TestCase):

    def test_apply_prediction_to_iterator(self):
        if self.multi_pred_values:
            def predict(imgs):
                n_img = len(imgs)
                return (
                    [np.random.uniform(size=(10, 4)) for _ in range(n_img)],
                    [np.random.uniform(size=10) for _ in range(n_img)],
                    [np.random.uniform(size=10) for _ in range(n_img)])

            n_pred_values = 3
        else:
            def predict(imgs):
                n_img = len(imgs)
                return [np.random.uniform(size=(48, 64)) for _ in range(n_img)]

            n_pred_values = 1

        dataset_imgs = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            dataset_imgs.append(np.random.randint(0, 256, size=(3, H, W)))

        if self.with_gt_values:
            strs = ['a', 'bc', 'def', 'ghij', 'klmno']
            nums = [0, 1, 2, 3, 4]
            arrays = [np.random.uniform(size=10) for _ in range(5)]

            dataset = chainer.datasets.TupleDataset(
                dataset_imgs, strs, nums, arrays)
            dataset_gt_values = (strs, nums, arrays)
        else:
            dataset = dataset_imgs
            dataset_gt_values = tuple()
        iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)

        if self.with_hook:
            def hook(imgs, pred_values, gt_values):
                self.assertEqual(len(pred_values), n_pred_values)
                for pred_vals in pred_values:
                    self.assertEqual(len(pred_vals), len(imgs))

                self.assertEqual(len(gt_values), len(dataset_gt_values))
                for gt_vals in gt_values:
                    self.assertEqual(len(gt_vals), len(imgs))
        else:
            hook = None

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            predict, iterator, hook=hook)

        for img, dataset_img in zip_longest(imgs, dataset_imgs):
            np.testing.assert_equal(img, dataset_img)

        self.assertEqual(len(pred_values), n_pred_values)
        for vals in pred_values:
            self.assertEqual(len(list(vals)), len(dataset_imgs))

        for vals, dataset_vals in zip_longest(gt_values, dataset_gt_values):
            for val, dataset_val in zip_longest(vals, dataset_vals):
                if isinstance(dataset_val, np.ndarray):
                    np.testing.assert_equal(val, dataset_val)
                else:
                    self.assertEqual(val, dataset_val)


class TestApplyPredictionToIteratorWithInfiniteIterator(unittest.TestCase):

    def test_apply_prediction_to_iterator_with_infinite_iterator(self):
        def predict(imgs):
            n_img = len(imgs)
            return [np.random.uniform(size=(48, 64)) for _ in range(n_img)]

        dataset = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            dataset.append(np.random.randint(0, 256, size=(3, H, W)))

        iterator = SerialIterator(dataset, 2)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            predict, iterator)

        for _ in range(10):
            next(imgs)

        for _ in range(10):
            next(pred_values[0])


testing.run_module(__name__, __file__)
