from __future__ import division

import unittest

import numpy as np

from chainer import testing

from chainercv.links.model.faster_rcnn import generate_anchor_base


class TestGenerateAnchorBase(unittest.TestCase):

    def test_generaete_anchor_base(self):
        gt = np.array(
            [[-120., -24., 136., 40.],
             [-248., -56., 264., 72.],
             [-504., -120., 520., 136.],
             [-56., -56., 72., 72.],
             [-120., -120., 136., 136.],
             [-248., -248., 264., 264.],
             [-24., -120., 40., 136.],
             [-56., -248., 72., 264.],
             [-120., -504., 136., 520.]])

        base_size = 16
        scales = [8, 16, 32]
        ratios = [0.25, 1, 4]
        out = generate_anchor_base(base_size=base_size,
                                   scales=scales,
                                   ratios=ratios)
        np.testing.assert_equal(gt, out)


testing.run_module(__name__, __file__)
