import numpy as np
import six

from chainercv.utils.testing.assertions.assert_is_bbox import assert_is_bbox
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


def assert_is_detection_dataset(dataset, n_fg_class, n_example=None):
    """Checks if a dataset satisfies detection dataset APIs.

    This function checks if a given dataset satisfies detection dataset APIs
    or not.
    If the dataset does not satifiy the APIs, this function raises an
    :class:`AssertionError`.

    Args:
        dataset: A dataset to be checked.
        n_fg_class (int): The number of foreground classes.
        n_example (int): The number of examples to be checked.
            If this argument is specified, this function picks
            examples ramdomly and checks them. Otherwise,
            this function checks all examples.

    """

    assert len(dataset) > 0, 'The length of dataset must be greater than zero.'

    if n_example:
        for _ in six.moves.range(n_example):
            i = np.random.randint(0, len(dataset))
            _check_example(dataset[i], n_fg_class)
    else:
        for i in six.moves.range(len(dataset)):
            _check_example(dataset[i], n_fg_class)


def _check_example(example, n_fg_class):
    assert len(example) >= 3, \
        'Each example must have at least three elements:' \
        'img, bbox and label.'

    img, bbox, label = example[:3]

    assert_is_image(img, color=True)
    assert_is_bbox(bbox, size=img.shape[1:])

    assert isinstance(label, np.ndarray), \
        'label must be a numpy.ndarray.'
    assert label.dtype == np.int32, \
        'The type of label must be numpy.int32.'
    assert label.shape[1:] == (), \
        'The shape of label must be (*,).'
    assert len(label) == len(bbox), \
        'The length of label must be same as that of bbox.'
    assert label.min() >= 0 and label.max() < n_fg_class, \
        'The value of label must be in [0, n_fg_class - 1].'
