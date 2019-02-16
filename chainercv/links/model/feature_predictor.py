import numpy as np
import warnings

import chainer
from chainer.backends import cuda

from chainercv.transforms import center_crop
from chainercv.transforms import resize
from chainercv.transforms import scale
from chainercv.transforms import ten_crop


class FeaturePredictor(chainer.Chain):

    """Wrapper that adds a prediction method to a feature extraction link.

    The :meth:`predict` takes three steps to make a prediction.

    1. Preprocess input images
    2. Forward the preprocessed images to the network
    3. Average features in the case when more than one crops are extracted.

    Example:

        >>> from chainercv.links import VGG16
        >>> from chainercv.links import FeaturePredictor
        >>> base_model = VGG16()
        >>> model = FeaturePredictor(base_model, 224, 256)
        >>> prob = model.predict([img])
        # Predicting multiple features
        >>> model.extractor.pick = ['conv5_3', 'fc7']
        >>> conv5_3, fc7 = model.predict([img])

    When :obj:`self.crop == 'center'`, :meth:`predict` extracts features from
    the center crop of the input images.
    When :obj:`self.crop == '10'`, :meth:`predict` extracts features from
    patches that are ten-cropped from the input images.

    When extracting more than one crops from an image, the output of
    :meth:`predict` returns the average of the features computed from the
    crops.

    Args:
        extractor: A feature extraction link. This is a callable chain
            that takes a batch of images and returns a variable or a
            tuple of variables.
        crop_size (int or tuple): The height and the width of an image after
            cropping in preprocessing.
            If this is an integer, the image is cropped to
            :math:`(crop\_size, crop\_size)`.
        scale_size (int or tuple): If :obj:`scale_size` is :obj:`None`,
            neither scaling nor resizing is conducted during preprocessing.
            This is the default behavior.
            If this is an integer, an image is resized so that the length of
            the shorter edge is equal to :obj:`scale_size`. If this is a tuple
            :obj:`(height, width)`, the image is resized to
            :math:`(height, width)`.
        crop ({'center', '10'}): Determines the style of cropping.
        mean (numpy.ndarray): A mean value. If this is :obj:`None`,
            :obj:`extractor.mean` is used as the mean value.

    """

    def __init__(self, extractor,
                 crop_size, scale_size=None,
                 crop='center', mean=None):
        super(FeaturePredictor, self).__init__()
        self.scale_size = scale_size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.crop = crop
        with self.init_scope():
            self.extractor = extractor

        if mean is None:
            self.mean = self.extractor.mean
        else:
            self.mean = mean

    def _prepare(self, img):
        """Prepare an image for feeding it to a model.

        This is a standard preprocessing scheme used by feature extraction
        models.
        First, the image is scaled or resized according to :math:`scale_size`.
        Note that this step is optional.
        Next, the image is cropped to :math:`crop_size`.
        Last, the image is mean subtracted by an array :obj:`mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image. This is 4D array whose batch size is
            the number of crops.

        """
        if self.scale_size is not None:
            if isinstance(self.scale_size, int):
                img = scale(img, size=self.scale_size)
            else:
                img = resize(img, size=self.scale_size)
        else:
            img = img.copy()

        if self.crop == '10':
            imgs = ten_crop(img, self.crop_size)
        elif self.crop == 'center':
            imgs = center_crop(img, self.crop_size)[np.newaxis]

        imgs -= self.mean[np.newaxis]

        return imgs

    def _average_crops(self, y, n_crop):
        if y.ndim == 4:
            warnings.warn(
                'Four dimensional features are averaged. '
                'If these are batch of 2D spatial features, '
                'their spatial information would be lost.')

        xp = chainer.backends.cuda.get_array_module(y)
        y = y.reshape((-1, n_crop) + y.shape[1:])
        y = xp.mean(y, axis=1)
        return y

    def predict(self, imgs):
        """Predict features from images.

        Given :math:`N` input images, this method outputs a batched array with
        batchsize :math:`N`.

        Args:
            imgs (iterable of numpy.ndarray): Array-images.
                All images are in CHW format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            numpy.ndarray or tuple of numpy.ndarray:
            A batch of features or a tuple of them.

        """
        # [(C, H_0, W_0), ..., (C, H_{B-1}, W_{B-1})] -> (B, N, C, H, W)
        imgs = self.xp.asarray([self._prepare(img) for img in imgs])
        n_crop = imgs.shape[-4]
        shape = (-1, imgs.shape[-3]) + self.crop_size
        # (B, N, C, H, W) -> (B * N, C, H, W)
        imgs = imgs.reshape(shape)

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            features = self.extractor(imgs)

        if isinstance(features, tuple):
            output = []
            for feature in features:
                feature = feature.array
                if n_crop > 1:
                    feature = self._average_crops(feature, n_crop)
                output.append(cuda.to_cpu(feature))
            output = tuple(output)
        else:
            output = cuda.to_cpu(features.array)
            if n_crop > 1:
                output = self._average_crops(output, n_crop)

        return output
