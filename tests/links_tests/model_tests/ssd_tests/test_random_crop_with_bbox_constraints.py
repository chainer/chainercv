from __future__ import division

import numpy as np
import unittest

from chainer import testing

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.utils import bbox_iou
from chainercv.utils import generate_random_bbox


class TestRandomCropWithBboxConstraints(unittest.TestCase):

    def test_random_crop_with_bbox_constraints(self):
        img = np.random.randint(0, 256, size=(3, 480, 640)).astype(np.float32)
        bbox = generate_random_bbox(10, img.shape[1:], 0.1, 0.9)

        out, param = random_crop_with_bbox_constraints(
            img, bbox,
            min_scale=0.3, max_scale=1,
            max_aspect_ratio=2,
            return_param=True)

        if param['constraint'] is None:
            np.testing.assert_equal(out, img)
        else:
            np.testing.assert_equal(
                out, img[:, param['y_slice'], param['x_slice']])

            # to ignore rounding error, add 1
            self.assertGreaterEqual(
                out.shape[0] * (out.shape[1] + 1) * (out.shape[2] + 1),
                img.size * 0.3 * 0.3)
            self.assertLessEqual(out.size, img.size * 1 * 1)
            self.assertLessEqual(
                out.shape[1] / (out.shape[2] + 1),
                img.shape[1] / img.shape[2] * 2)
            self.assertLessEqual(
                out.shape[2] / (out.shape[1] + 1),
                img.shape[2] / img.shape[1] * 2)

            bb = np.array((
                param['y_slice'].start, param['x_slice'].start,
                param['y_slice'].stop, param['x_slice'].stop))
            iou = bbox_iou(bb[np.newaxis], bbox)
            min_iou, max_iou = param['constraint']
            if min_iou:
                self.assertGreaterEqual(iou.min(), min_iou)
            if max_iou:
                self.assertLessEqual(iou.max(), max_iou)


testing.run_module(__name__, __file__)
