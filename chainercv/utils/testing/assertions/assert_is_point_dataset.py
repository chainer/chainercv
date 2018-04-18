import numpy as np
import six

from chainercv.utils.testing.assertions.assert_is_image import assert_is_image
from chainercv.utils.testing.assertions.assert_is_point import assert_is_point


def assert_is_point_dataset(dataset, n_point=None, n_example=None,
                            no_mask=False):
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
        no_mask (bool): If :obj:`True`, we assume that
            point mask is always not contained.
            If :obj:`False`, point mask may or may not be contained.

    """

    assert len(dataset) > 0, 'The length of dataset must be greater than zero.'

    if n_example:
        for _ in six.moves.range(n_example):
            i = np.random.randint(0, len(dataset))
            _check_example(dataset[i], n_point, no_mask)
    else:
        for i in six.moves.range(len(dataset)):
            _check_example(dataset[i], n_point, no_mask)


def _check_example(example, n_point=None, no_mask=False):
    assert len(example) >= 2, \
        'Each example must have at least two elements:' \
        'img, point (mask is optional).'

    if len(example) == 2 or no_mask:
        img, point = example[:2]
        mask = None
    elif len(example) >= 3:
        img, point, mask = example[:3]

    assert_is_image(img, color=True)
    assert_is_point(point, mask, img.shape[1:])

    if n_point is not None:
        assert point.shape[0] == n_point, \
            'The number of points is different from the expected number.'
