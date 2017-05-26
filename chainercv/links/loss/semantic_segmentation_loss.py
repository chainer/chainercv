import chainer
from chainer import cuda
import chainer.functions as F
from chainer import reporter

import numpy as np

from chainercv.evaluations import eval_semantic_segmentation


def _segmentation_accuracies(y, t, n_class):
    y = F.argmax(F.softmax(y), axis=1)
    if y.ndim == 3:
        y = y[:, None, :, :]
    if t.ndim == 3:
        t = t[:, None, :, :]
    xp = cuda.get_array_module(y.data, t.data)
    return [xp.mean(ret)
            for ret in eval_semantic_segmentation(y.data, t.data, n_class)]


class PixelwiseSigmoidClassifier(chainer.Chain):

    def __init__(self, model, n_class, compute_accuracy=True):
        super(PixelwiseSigmoidLoss, self).__init__(predictor=model)
        self.n_class = n_class
        self.compute_accuracy = compute_accuracy

    def __call__(self, x, t):
        self.y = self.predictor(x)
        self.loss = F.sigmoid_cross_entropy(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            pa, mpa, miou, fwiou = _segmentation_accuracies(
                self.y, t, self.n_class)
            reporter.report({
                'pixel_accuracy': pa,
                'mean_pixel_accuracy': mpa,
                'mean_iou': miou,
                'frequency_weighted_iou': fwiou
            }, self)
        return self.loss


class PixelwiseSoftmaxClassifier(chainer.Chain):

    def __init__(self, model, n_class, ignore_label=-1, compute_accuracy=True):
        super(PixelwiseSoftmaxLoss, self).__init__(predictor=model)
        self.n_class = n_class
        self.ignore_label = ignore_label
        self.compute_accuracy = compute_accuracy

    def __call__(self, x, t):
        self.y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(
            self.y, t, ignore_label=self.ignore_label)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            pa, mpa, miou, fwiou = _segmentation_accuracies(
                self.y, t, self.n_class)
            reporter.report({
                'pixel_accuracy': pa,
                'mean_pixel_accuracy': mpa,
                'mean_iou': miou,
                'frequency_weighted_iou': fwiou
            }, self)
        return self.loss


class PixelwiseSoftmaxWithWeightClassifier(chainer.Chain):

    def __init__(self, model, n_class, ignore_label=-1, class_weight=None,
                 compute_accuracy=True):
        super(PixelwiseSoftmaxLossWithWeight, self).__init__(predictor=model)
        self.n_class = n_class
        self.ignore_label = ignore_label
        self.class_weight = np.asarray(class_weight, dtype=np.float32)
        self.compute_accuracy = compute_accuracy

    def __call__(self, x, t):
        if not hasattr(self.class_weight, 'device'):
            self.class_weight = cuda.to_gpu(self.class_weight, x.data.device)
        self.y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(
            self.y, t, class_weight=self.class_weight,
            ignore_label=self.ignore_label)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            pa, mpa, miou, fwiou = _segmentation_accuracies(
                self.y, t, self.n_class)
            reporter.report({
                'pixel_accuracy': pa,
                'mean_pixel_accuracy': mpa,
                'mean_iou': miou,
                'frequency_weighted_iou': fwiou
            }, self)
        return self.loss
