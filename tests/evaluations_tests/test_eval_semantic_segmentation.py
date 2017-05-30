import unittest

import numpy as np

from chainer import testing

from chainercv.evaluations import calc_confusion_matrix
from chainercv.evaluations import eval_semantic_segmentation


@testing.parameterize(
    {'pred_labels': np.repeat([[[1, 1, 0], [0, 0, 1]]], 2, axis=0),
     'gt_labels': np.repeat([[[1, 0, 0], [0, -1, 1]]], 2, axis=0),
     'iou': np.array([4. / 6., 4. / 6.])
     },
    {'pred_labels': [np.array([[1, 1, 0], [0, 0, 1]]),
                     np.array([[1, 1, 0], [0, 0, 1]])],
     'gt_labels': [np.array([[1, 0, 0], [0, -1, 1]]),
                   np.array([[1, 0, 0], [0, -1, 1]])],
     'iou': np.array([4. / 6., 4. / 6.])
     },
    {'pred_labels': np.array([[[0, 0, 0], [0, 0, 0]]]),
     'gt_labels': np.array([[[1, 1, 1], [1, 1, 1]]]),
     'iou': np.array([0, 0]),
     }
)
class TestEvalSemanticSegmentation(unittest.TestCase):

    n_class = 2

    def test_calc_confusion_matrix(self):

        iou = eval_semantic_segmentation(
            self.pred_labels, self.gt_labels, self.n_class)
        np.testing.assert_equal(iou, self.iou)


@testing.parameterize(
    {'pred_label': np.array([[1, 1, 0], [0, 0, 1]]),
     'gt_label': np.array([[1, 0, 0], [0, -1, 1]]),
     'confusion': np.array([[2, 1], [0, 2]])
     },
    {'pred_label': np.array([[0, 0, 0], [0, 0, 0]]),
     'gt_label': np.array([[1, 1, 1], [1, 1, -1]]),
     'confusion': np.array([[0, 0], [5, 0]])
     }
)
class TestCalcConfusionMatrix(unittest.TestCase):

    n_class = 2

    def test_calc_confusion_matrix(self):
        confusion = calc_confusion_matrix(
            self.pred_label, self.gt_label, self.n_class)
        np.testing.assert_equal(confusion, self.confusion)


testing.run_module(__name__, __file__)
