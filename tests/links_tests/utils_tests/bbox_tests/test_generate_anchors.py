from __future__ import division

import unittest

import numpy as np

from chainer import testing

from chainercv.links.utils.bbox.generate_anchors import generate_anchors


class TestGenerateAnchors(unittest.TestCase):

    def test_generaete_anchors(self):
        base_size = 16
        scales = [8, 16]
        ratios = [0.25, 4]
        x = generate_anchors(base_size=base_size,
                             scales=scales,
                             ratios=ratios)
        # Center coordinates of anchors
        px = base_size / 2
        py = base_size / 2

        for i in range(len(ratios)):
            for j in range(len(scales)):
                w = base_size * scales[j] * np.sqrt(1. / ratios[i])
                h = base_size * scales[j] * np.sqrt(ratios[i])

                index = i * len(scales) + j
                np.testing.assert_equal(x[index, 0], px - w / 2.)
                np.testing.assert_equal(x[index, 1], py - h / 2.)
                np.testing.assert_equal(x[index, 2], px + w / 2.)
                np.testing.assert_equal(x[index, 3], py + h / 2.)


testing.run_module(__name__, __file__)
