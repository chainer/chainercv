import unittest

import numpy as np

from chainer import testing

from chainercv.evaluations import calc_confusion_matrix
from chainercv.evaluations import eval_semantic_segmentation


@testing.parameterize(
    {'pred_labels': np.repeat([[[1, 1, 0], [0, 0, 1]]], 2, axis=0),
     'gt_labels': np.repeat([[[1, 0, 0], [0, -1, 1]]], 2, axis=0),
     'confusion': np.array([[4, 2], [0, 4]])
     },
    {'pred_labels': [np.array([[1, 1, 0], [0, 0, 1]]),
                     np.array([[1, 1, 0], [0, 0, 1]])],
     'gt_labels': [np.array([[1, 0, 0], [0, -1, 1]]),
                   np.array([[1, 0, 0], [0, -1, 1]])],
     'confusion': np.array([[4, 2], [0, 4]])
     },
    {'pred_labels': np.array([[[0, 0, 0], [0, 0, 0]]]),
     'gt_labels': np.array([[[1, 1, 1], [1, 1, 1]]]),
     'confusion': np.array([[0., 0], [6, 0]]),
     }
)
class TestCalcConfusionMatrix(unittest.TestCase):

    n_class = 2

    def test_calc_confusion_matrix(self):
        confusion = calc_confusion_matrix(
            self.pred_labels, self.gt_labels, self.n_class)

        self.assertIsInstance(confusion, np.ndarray)
        np.testing.assert_equal(confusion, self.confusion)


@testing.parameterize(
    {'confusion': np.array([[4, 2], [0, 4]]),
     'iou': np.array([4. / 6., 4. / 6.])
     },
    {'confusion': np.array([[0, 0], [6, 0]]),
     'iou': np.array([0, 0]),
     }
)
class TestEvalSemanticSegmentation(unittest.TestCase):

    def test_eval_semantic_segmentation(self):
        iou = eval_semantic_segmentation(self.confusion)
        np.testing.assert_equal(iou, self.iou)


testing.run_module(__name__, __file__)
