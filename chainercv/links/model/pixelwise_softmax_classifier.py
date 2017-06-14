import chainer
from chainer import cuda
import chainer.functions as F
from chainer import reporter

import numpy as np


class PixelwiseSoftmaxClassifier(chainer.Chain):

    """A pixel-wise classifier.

    It computes the loss based on a given input/label pair for
    semantic segmentation.

    Args:
        predictor (~chainer.Link): Predictor network.
        ignore_label (int): A class id that is going to be ignored in
            evaluation. The default value is -1.
        class_weight (array): An array
            that contains constant weights that will be multiplied with the
            loss values along with the channel dimension. This will be
            used in :func:`chainer.functions.softmax_cross_entropy`.

    """

    def __init__(self, predictor, ignore_label=-1, class_weight=None):
        super(PixelwiseSoftmaxClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.n_class = predictor.n_class
        self.ignore_label = ignore_label
        if class_weight is not None:
            self.class_weight = np.asarray(class_weight, dtype=np.float32)
        else:
            self.class_weight = class_weight

    def to_cpu(self):
        super(PixelwiseSoftmaxClassifier, self).to_cpu()
        if self.class_weight is not None:
            self.class_weight = cuda.to_cpu(self.class_weight)

    def to_gpu(self, device=None):
        super(PixelwiseSoftmaxClassifier, self).to_gpu(device)
        if self.class_weight is not None:
            self.class_weight = cuda.to_gpu(self.class_weight, device)

    def __call__(self, x, t):
        """Computes the loss value for an image and label pair.

        Args:
            x (~chainer.Variable): A variable with a batch of images.
            t (~chainer.Variable): A variable with the ground truth
                image-wise label.

        Returns:
            ~chainer.Variable: Loss value.

        """
        self.y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(
            self.y, t, class_weight=self.class_weight,
            ignore_label=self.ignore_label)

        reporter.report({'loss': self.loss}, self)
        return self.loss
