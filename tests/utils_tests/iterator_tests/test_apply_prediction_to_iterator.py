import numpy as np
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
class TestApplyDetectionLink(unittest.TestCase):

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

        self.assertEqual(list(imgs), dataset_imgs)

        self.assertEqual(len(pred_values), n_pred_values)
        for pred_vals in pred_values:
            self.assertEqual(len(list(pred_vals)), len(dataset_imgs))

        self.assertEqual(len(gt_values), len(dataset_gt_values))
        for gt_vals, dataset_gt_vals in zip(gt_values, dataset_gt_values):
            self.assertEqual(list(gt_vals), dataset_gt_vals)


testing.run_module(__name__, __file__)
