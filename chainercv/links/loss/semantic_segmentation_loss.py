import chainer
from chainer import cuda
import chainer.functions as F
from chainer import reporter

import numpy as np

from chainercv.evaluations import eval_semantic_segmentation


class PixelwiseSoftmaxClassifier(chainer.Chain):

    def __init__(self, model, ignore_label=-1, class_weight=None,
                 compute_accuracy=True):
        super(PixelwiseSoftmaxClassifier, self).__init__(predictor=model)
        self.n_class = model.n_class
        self.ignore_label = ignore_label
        if class_weight is not None:
            self.class_weight = np.asarray(class_weight, dtype=np.float32)
        else:
            self.class_weight = class_weight
        self.compute_accuracy = compute_accuracy

    def to_cpu(self):
        super(PixelwiseSoftmaxClassifier, self).to_cpu()
        if self.class_weight is not None:
            self.class_weight = cuda.to_cpu(self.class_weight)

    def to_gpu(self):
        super(PixelwiseSoftmaxClassifier, self).to_gpu()
        if self.class_weight is not None:
            self.class_weight = cuda.to_gpu(self.class_weight)

    def __call__(self, x, t):
        self.y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(
            self.y, t, class_weight=self.class_weight,
            ignore_label=self.ignore_label)

        reporter.report({'loss': self.loss}, self)

        self.accuracy = None
        if self.compute_accuracy:
            label = self.xp.argmax(self.y.data, axis=1)
            self.accuracy = eval_semantic_segmentation(
                label, t.data, self.n_class)
            reporter.report({
                'pixel_accuracy': self.xp.mean(self.accuracy[0]),
                'mean_pixel_accuracy': self.xp.mean(self.accuracy[1]),
                'mean_iou': self.xp.mean(self.accuracy[2]),
                'frequency_weighted_iou': self.xp.mean(self.accuracy[3])
            }, self)
        return self.loss
