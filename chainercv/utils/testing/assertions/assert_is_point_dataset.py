import numpy as np
import six

from chainercv.utils.testing.assertions.assert_is_image import assert_is_image
from chainercv.utils.testing.assertions.assert_is_point import assert_is_point


def assert_is_point_dataset(dataset, n_point=None, n_example=None,
                            no_visible=False):
    """Checks if a dataset satisfies the point dataset API.

    This function checks if a given dataset satisfies the point dataset
    API or not.
    If the dataset does not satifiy the API, this function raises an
    :class:`AssertionError`.

    Args:
        dataset: A dataset to be checked.
        n_point (int): The number of expected points per image.
            If this is :obj:`None`, the number of points per image can be
            arbitrary.
        n_example (int): The number of examples to be checked.
            If this argument is specified, this function picks
            examples ramdomly and checks them. Otherwise,
            this function checks all examples.
        no_visible (bool): If :obj:`True`, we assume that
            visibility mask is always not contained.
            If :obj:`False`, point visible may or may not be contained.

    """

    assert len(dataset) > 0, 'The length of dataset must be greater than zero.'

    if n_example:
        for _ in six.moves.range(n_example):
            i = np.random.randint(0, len(dataset))
            _check_example(dataset[i], n_point, no_visible)
    else:
        for i in six.moves.range(len(dataset)):
            _check_example(dataset[i], n_point, no_visible)


def _check_example(example, n_point=None, no_visible=False):
    assert len(example) >= 2, \
        'Each example must have at least two elements:' \
        'img, point (visible is optional).'

    if len(example) == 2 or no_visible:
        img, point = example[:2]
        visible = None
    elif len(example) >= 3:
        img, point, visible = example[:3]

    assert_is_image(img, color=True)
    assert_is_point(point, visible, img.shape[1:], n_point)
