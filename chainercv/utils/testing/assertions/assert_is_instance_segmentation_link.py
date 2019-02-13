import numpy as np
import six


def assert_is_instance_segmentation_link(link, n_fg_class):
    """Checks if a link satisfies instance segmentation link APIs.

    This function checks if a given link satisfies instance segmentation
    link APIs or not.
    If the link does not satifiy the APIs, this function raises an
    :class:`AssertionError`.

    Args:
        link: A link to be checked.
        n_fg_class (int): The number of foreground classes.

    """

    imgs = [
        np.random.randint(0, 256, size=(3, 480, 640)).astype(np.float32),
        np.random.randint(0, 256, size=(3, 480, 320)).astype(np.float32)]

    result = link.predict(imgs)
    assert len(result) == 3, \
        'Link must return three elements: masks, labels and scores.'
    masks, labels, scores = result

    assert len(masks) == len(imgs), \
        'The length of masks must be same as that of imgs.'
    assert len(labels) == len(imgs), \
        'The length of labels must be same as that of imgs.'
    assert len(scores) == len(imgs), \
        'The length of scores must be same as that of imgs.'

    for img, mask, label, score in six.moves.zip(imgs, masks, labels, scores):

        assert isinstance(mask, np.ndarray), \
            'mask must be a numpy.ndarray.'
        assert mask.dtype == np.bool, \
            'The type of mask must be bool'
        assert mask.shape[1:] == img.shape[1:], \
            'The shape of mask must be (R, H, W).'

        assert isinstance(label, np.ndarray), \
            'label must be a numpy.ndarray.'
        assert label.dtype == np.int32, \
            'The type of label must be numpy.int32.'
        assert label.shape[1:] == (), \
            'The shape of label must be (*,).'
        assert len(label) == len(mask), \
            'The length of label must be same as that of mask.'
        if len(label) > 0:
            assert label.min() >= 0 and label.max() < n_fg_class, \
                'The value of label must be in [0, n_fg_class - 1].'

        assert isinstance(score, np.ndarray), \
            'score must be a numpy.ndarray.'
        assert score.dtype == np.float32, \
            'The type of score must be numpy.float32.'
        assert score.shape[1:] == (), \
            'The shape of score must be (*,).'
        assert len(score) == len(mask), \
            'The length of score must be same as that of mask.'
