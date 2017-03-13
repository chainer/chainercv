import unittest

import numpy as np

from chainer import testing

from chainercv.tasks import eval_pck


class TestEvalPCK(unittest.TestCase):

    def test_eval_pck(self):
        pred = np.array([[0, 0], [1, 1]], dtype=np.int32)
        expected = np.array([[0, 0], [1, 2]], dtype=np.int32)

        pck_pred = eval_pck(pred, expected, alpha=1., L=2.)
        self.assertAlmostEqual(pck_pred, 1.)

        pck_pred = eval_pck(pred, expected, alpha=0.25, L=2.)
        self.assertAlmostEqual(pck_pred, 0.5)

    def test_eval_pck_ndim_3(self):
        pred = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32)
        expected = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int32)

        pck_pred = eval_pck(pred, expected, alpha=1., L=2.)
        self.assertAlmostEqual(pck_pred, 1.)

        pck_pred = eval_pck(pred, expected, alpha=0.25, L=2.)
        self.assertAlmostEqual(pck_pred, 1.)


testing.run_module(__name__, __file__)
