import unittest

import numpy as np

from chainercv.evaluations import eval_semantic_segmentation


class TestEvalSemanticSegmentation(unittest.TestCase):

    def test_eval_semantic_segmentation(self):
        predict = np.array([[1, 1, 0], [0, 0, 1]]).reshape(1, 2, 3)
        gt = np.array([[1, 0, 0], [0, -1, 1]]).reshape(1, 2, 3)
        # p_00 = 2
        # p_01 = 1
        # p_10 = 0
        # p_11 = 2
        acc, acc_cls, mean_iu, fwavacc = eval_semantic_segmentation(
            predict, gt, n_class=2)

        self.assertTrue(isinstance(acc, np.ndarray))
        self.assertTrue(isinstance(acc_cls, np.ndarray))
        self.assertTrue(isinstance(mean_iu, np.ndarray))
        self.assertTrue(isinstance(fwavacc, np.ndarray))
        self.assertEqual(acc[0], 4. / 5.)
        self.assertEqual(acc_cls[0], 1. / 2. * (1 + 2. / 3.))
        self.assertEqual(mean_iu[0], 1. / 2. * (1. / 3. + 1))
        self.assertEqual(fwavacc[0], 1. / 5. * (2. + 4. / 3.))

    def test_eval_semantic_segmentation_batch(self):
        predict = np.array([[1, 1, 0], [0, 0, 1]]).reshape(1, 2, 3)
        predict = np.stack((predict, predict))
        gt = np.array([[1, 0, 0], [0, -1, 1]]).reshape(1, 2, 3)
        gt = np.stack((gt, gt))
        # p_00 = 2
        # p_01 = 1
        # p_10 = 0
        # p_11 = 2
        acc, acc_cls, mean_iu, fwavacc = eval_semantic_segmentation(
            predict, gt, n_class=2)

        self.assertTrue(isinstance(acc, np.ndarray))
        self.assertTrue(isinstance(acc_cls, np.ndarray))
        self.assertTrue(isinstance(mean_iu, np.ndarray))
        self.assertTrue(isinstance(fwavacc, np.ndarray))
        for i in range(2):
            self.assertEqual(acc[i], 4. / 5.)
            self.assertEqual(acc_cls[i], 1. / 2. * (1 + 2. / 3.))
            self.assertEqual(mean_iu[i], 1. / 2. * (1. / 3. + 1))
            self.assertEqual(fwavacc[i], 1. / 5. * (2. + 4. / 3.))
