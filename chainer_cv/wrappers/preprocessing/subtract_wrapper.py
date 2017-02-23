import numpy as np

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


vgg_subtract_bgr = np.array([103.939, 116.779, 123.68], np.float32)


class SubtractWrapper(DatasetWrapper):

    """Subtract images by constants

    Subtract values from the first array returned by the wrapped dataset.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        value (float or array-like): constant to subtract data from. This is
            either 0 or 1 dimensional array. The value is subtracted along
            channel axis. Default is the constant value used to subtract
            from image when preparing for computer vision networks like VGG.

    """

    def __init__(self, dataset, value=vgg_subtract_bgr):
        super(SubtractWrapper, self).__init__(dataset)
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        # convert to CHW format
        if value.ndim == 0:
            value = value[None, None, None]
        elif value.ndim == 1:
            value = value[:, None, None]
        elif value.ndim != 3:
            raise ValueError('only accept 0d, 1d or 3d array')

        self.value = value

    def _get_example(self, in_data):
        x = in_data[0]
        x -= self.value
        return (x,) + in_data[1:]
