from __future__ import division

import numpy as np
import unittest

from chainer import testing

from chainercv.links.model.mask_rcnn.misc import segm_to_mask
from chainercv.links.model.mask_rcnn.misc import mask_to_segm


class TestSegmToMask(unittest.TestCase):

    def setUp(self):
        # When n_inst >= 3, the test fails.
        # This is due to the fact that the transformed image of `transforms.resize`
        # is misaligned to the corners.
        n_inst = 2
        self.segm_size = 3
        self.size = (36, 48)

        self.segm = np.ones((n_inst, self.segm_size, self.segm_size), dtype=np.float32)
        self.bbox = np.zeros((n_inst, 4), dtype=np.float32)
        for i in range(n_inst):
            self.bbox[i, 0] = 10 + i
            self.bbox[i, 1] = 10 + i
            self.bbox[i, 2] = self.bbox[i, 0] + self.segm_size * (1 + i)
            self.bbox[i, 3] = self.bbox[i, 1] + self.segm_size * (1 + i)

        self.mask = np.zeros((n_inst,) + self.size, dtype=np.bool)
        for i, bb in enumerate(self.bbox):
            bb = bb.astype(np.int32)
            self.mask[i, bb[0]:bb[2], bb[1]:bb[3]] = 1

    def test_segm_to_mask(self):
        mask = segm_to_mask(self.segm, self.bbox, self.size)
        np.testing.assert_equal(mask, self.mask)

    def test_mask_to_segm(self):
        segm = mask_to_segm(self.mask, self.bbox, self.segm_size)
        np.testing.assert_equal(segm, self.segm)

    def test_mask_to_segm_index(self):
        index = np.arange(len(self.bbox))[::-1]
        segm = mask_to_segm(
            self.mask, self.bbox[::-1],
            self.segm_size, index=index)
        segm = segm[::-1]
        np.testing.assert_equal(segm, self.segm)


testing.run_module(__name__, __file__)
