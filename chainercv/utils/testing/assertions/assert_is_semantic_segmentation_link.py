import numpy as np
import six


def assert_is_semantic_segmentation_link(link, n_class):
    """Checks if a link satisfies semantic segmentation link APIs.

    This function checks if a given link satisfies semantic segmentation link
    APIs or not.
    If the link does not satifiy the APIs, this function raises an
    :class:`AssertionError`.

    Args:
        link: A link to be checked.
        n_class (int): The number of classes including background.

    """

    imgs = [
        np.random.randint(0, 256, size=(3, 480, 640)).astype(np.float32),
        np.random.randint(0, 256, size=(3, 480, 320)).astype(np.float32)]

    labels = link.predict(imgs)
    assert len(labels) == len(imgs), \
        'The length of labels must be same as that of imgs.'

    for img, label in six.moves.zip(imgs, labels):
        assert isinstance(label, np.ndarray), \
            'label must be a numpy.ndarray.'
        assert label.dtype == np.int32, \
            'The type of label must be numpy.int32.'
        assert label.shape == img.shape[1:], \
            'The shape of label must be (H, W).'
        assert label.min() >= 0 and label.max() < n_class, \
            'The value of label must be in [0, n_class - 1].'
