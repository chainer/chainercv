import chainer
import chainer.functions as F
from chainer import reporter


class PixelwiseSoftmaxClassifier(chainer.Chain):

    """A pixel-wise classifier.

    It computes the loss based on a given input/label pair for
    semantic segmentation.

    Args:
        predictor (~chainer.Link): Predictor network.
        ignore_label (int): A class id that is going to be ignored in
            evaluation. The default value is -1.

    """

    def __init__(self, predictor, ignore_label=-1):
        super(PixelwiseSoftmaxClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.ignore_label = ignore_label

    def __call__(self, x, t):
        """Computes the loss value for an image and label pair.

        Args:
            x (~chainer.Variable): A variable with a batch of images.
            t (~chainer.Variable): A variable with the ground truth
                image-wise label.

        Returns:
            ~chainer.Variable: Loss value.

        """
        aux, y = self.predictor(x)
        aux_loss = F.softmax_cross_entropy(
            aux, t, ignore_label=self.ignore_label)
        main_loss = F.softmax_cross_entropy(
            y, t, ignore_label=self.ignore_label)
        del aux, y, t

        loss = aux_loss * 0.4 + main_loss
        reporter.report({'loss': loss}, self)
        return loss
