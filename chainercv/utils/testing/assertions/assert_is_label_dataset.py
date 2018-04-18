import numpy as np
import six

from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


def assert_is_label_dataset(dataset, n_class, n_example=None, color=True):
    """Checks if a dataset satisfies the label dataset API.

    This function checks if a given dataset satisfies the label dataset
    API or not.
    If the dataset does not satifiy the API, this function raises an
    :class:`AssertionError`.

    Args:
        dataset: A dataset to be checked.
        n_class (int): The number of classes.
        n_example (int): The number of examples to be checked.
            If this argument is specified, this function picks
            examples ramdomly and checks them. Otherwise,
            this function checks all examples.
        color (bool): A boolean that determines the expected channel size.
            If it is :obj:`True`, the number of channels
            should be :obj:`3`. Otherwise, it should be :obj:`1`.
            The default value is :obj:`True`.

    """

    assert len(dataset) > 0, 'The length of dataset must be greater than zero.'

    if n_example:
        for _ in six.moves.range(n_example):
            i = np.random.randint(0, len(dataset))
            _check_example(dataset[i], n_class, color)
    else:
        for i in six.moves.range(len(dataset)):
            _check_example(dataset[i], n_class, color)


def _check_example(example, n_class, color):
    assert len(example) >= 2, \
        'Each example must have at least two elements:' \
        'img and label.'

    img, label = example[:2]

    assert_is_image(img, color=color)

    assert isinstance(label, np.int32), \
        'label must be a numpy.int32.'
    assert label.ndim == 0, 'The ndim of label must be 0'
    assert label >= 0 and label < n_class, \
        'The value of label must be in [0, n_class - 1].'
