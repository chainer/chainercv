import numpy as np
import warnings

import chainer
from chainer import cuda

from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.transforms import ten_crop


class FeatureExtractionPredictor(chainer.Chain):

    """Wrapper class that adds predict method to a feature extraction model.

    The :meth:`predict` takes three steps to make predictions.

    1. Preprocess images
    2. Forward the preprocessed images to the network
    3. Average features in the case when ten-crop is used.

    Example:

        >>> from chainercv.links import VGG16
        >>> from chainercv.links import FeatureExtractionPredictor
        >>> base_model = VGG16()
        >>> model = FeatureExtractionPredictor(base_model)
        >>> prob = model.predict([img])

    """

    def __init__(self, extractor,
                 size=(224, 224), scale_size=256,
                 do_ten_crop=False):
        super(FeatureExtractionPredictor, self).__init__()
        self.scale_size = scale_size
        self.size = size
        self.do_ten_crop = do_ten_crop

        with self.init_scope():
            self.extractor = extractor

    @property
    def mean(self):
        return self.extractor.mean

    def _prepare(self, img):
        """Transform an image to the input for VGG network.

        This is a standard preprocessing scheme used by feature extraction
        models.
        First, the image is scaled so that the length of the smaller edge is
        :math:`scale_size`.
        Next, the image is center cropped or ten cropped to :math:`size`.
        Last, the image is mean subtracted by a mean image array :obj:`mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        img = scale(img, size=self.scale_size)
        if self.do_ten_crop:
            img = ten_crop(img, self.size)
            img -= self.mean[np.newaxis]
        else:
            img = center_crop(img, self.size)
            img -= self.mean

        return img

    def _average_ten_crop(self, y):
        if y.ndim == 4:
            warnings.warn(
                'Four dimensional features are averaged. '
                'If these are batch of 2D spatial features, '
                'their spatial information would be lost.')

        xp = chainer.cuda.get_array_module(y)
        n = y.shape[0] // 10
        y_shape = y.shape[1:]
        y = y.reshape((n, 10) + y_shape)
        y = xp.sum(y, axis=1) / 10
        return y

    def predict(self, imgs):
        """Predict features from images.

        When :obj:`self.do_ten_crop == True`, this extracts features from
        patches that are ten-cropped from images.
        Otherwise, this extracts features from center-crop of the images.

        When using patches from ten-crop, the features for each crop
        is averaged to compute one feature.

        Given :math:`N` input images, this outputs a batched array with
        batchsize :math:`N`.

        Args:
            imgs (iterable of numpy.ndarray): Array-images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            numpy.ndarray or tuple of numpy.ndarray:
            A batch of features or a tuple of them.

        """
        imgs = self.xp.asarray([self._prepare(img) for img in imgs])
        shape = (-1, imgs.shape[-3]) + self.size
        imgs = imgs.reshape(shape)

        with chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            activations = self.extractor(imgs)

        if isinstance(activations, tuple):
            output = []
            for activation in activations:
                activation = activation.data
                if self.do_ten_crop:
                    activation = self._average_ten_crop(activation)
                output.append(cuda.to_cpu(activation))
            output = tuple(output)
        else:
            output = cuda.to_cpu(activations.data)
            if self.do_ten_crop:
                output = self._average_ten_crop(output)

        return output
