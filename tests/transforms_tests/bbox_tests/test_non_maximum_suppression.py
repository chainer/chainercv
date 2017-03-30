import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import non_maximum_suppression


@testing.parameterize(
    {'threshold': 1, 'expect': (True, True, True, True)},
    {'threshold': 0.5, 'expect': (True, True, False, True)},
    {'threshold': 0.3, 'expect': (True, True, False, False)},
    {'threshold': 0.2, 'expect': (True, False, False, False)},
)
class TestNonMaximumSuppression(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array((
            (1, 2, 3, 4),
            (0, 1, 4, 5),  # 4/16
            (1, 2, 3, 5),  # 4/6, 6/16
            (1, 2, 5, 6),  # 6/16, 9/23, 4/16
        ))
        self.expect = np.array(self.expect)

    def test_non_maximum_suppression(self):
        out = non_maximum_suppression(self.bbox, self.threshold)
        np.testing.assert_equal(out, self.bbox[self.expect])

    def test_non_maximum_suppression_param(self):
        out, param = non_maximum_suppression(
            self.bbox, self.threshold, return_param=True)
        np.testing.assert_equal(param['selection'], self.expect)
        np.testing.assert_equal(out, self.bbox[param['selection']])


testing.run_module(__name__, __file__)
