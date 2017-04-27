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

        self.assertEqual(acc, 4. / 5.)
        self.assertEqual(acc_cls, 1. / 2. * (1 + 2. / 3.))
        self.assertEqual(mean_iu, 1. / 2. * (1. / 3. + 1))
        self.assertEqual(fwavacc, 1. / 5. * (2. + 4. / 3.))
