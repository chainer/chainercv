import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr


from chainercv.evaluations import eval_semantic_segmentation


@testing.parameterize(
    # p_00 = 2
    # p_01 = 1
    # p_10 = 0
    # p_11 = 2
    {'predict': [[[1, 1, 0], [0, 0, 1]]],
     'gt': [[[1, 0, 0], [0, -1, 1]]],
     'acc': [4. / 5.],
     'acc_cls': [1. / 2. * (1. + 2. / 3.)],
     'mean_iu': [1. / 2. * (1. / 3. + 1.)],
     'fwavacc': [1. / 5. * (2. + 4. / 3.)]
     },
    {'predict': np.repeat([[[[1, 1, 0], [0, 0, 1]]]], 2, axis=0),
     'gt': np.repeat([[[[1, 0, 0], [0, -1, 1]]]], 2, axis=0),
     'acc': [4. / 5., 4. / 5.],
     'acc_cls': [1. / 2. * (1. + 2. / 3.),
                 1. / 2. * (1. + 2. / 3.)],
     'mean_iu': [1. / 2. * (1. / 3. + 1.),
                 1. / 2. * (1. / 3. + 1.)],
     'fwavacc': [1. / 5. * (2. + 4. / 3.),
                 1. / 5. * (2. + 4. / 3.)]
     }
)
class TestEvalSemanticSegmentation(unittest.TestCase):

    n_class = 2

    def check_eval_semantic_segmentation(self, predict, gt, acc,
                                         acc_cls, mean_iu, fwavacc, n_class):
        # obtained
        acc_o, acc_cls_o, mean_iu_o, fwavacc_o = eval_semantic_segmentation(
            predict, gt, n_class=n_class)

        self.assertIsInstance(acc_o, type(acc))
        self.assertIsInstance(acc_cls_o, type(acc_cls))
        self.assertIsInstance(mean_iu_o, type(mean_iu))
        self.assertIsInstance(fwavacc_o, type(fwavacc))

        np.testing.assert_equal(cuda.to_cpu(acc_o), cuda.to_cpu(acc))
        np.testing.assert_equal(cuda.to_cpu(acc_cls_o), cuda.to_cpu(acc_cls))
        np.testing.assert_equal(cuda.to_cpu(mean_iu_o), cuda.to_cpu(mean_iu))
        np.testing.assert_equal(cuda.to_cpu(fwavacc_o), cuda.to_cpu(fwavacc))

    def test_eval_semantic_segmentation_cpu(self):
        self.check_eval_semantic_segmentation(
            np.array(self.predict),
            np.array(self.gt),
            np.array(self.acc),
            np.array(self.acc_cls),
            np.array(self.mean_iu),
            np.array(self.fwavacc),
            self.n_class)

    @attr.gpu
    def test_eval_semantic_segmentation_gpu(self):
        self.check_eval_semantic_segmentation(
            cuda.cupy.array(self.predict),
            cuda.cupy.array(self.gt),
            cuda.cupy.array(self.acc),
            cuda.cupy.array(self.acc_cls),
            cuda.cupy.array(self.mean_iu),
            cuda.cupy.array(self.fwavacc),
            self.n_class)


testing.run_module(__name__, __file__)
