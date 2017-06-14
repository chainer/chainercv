import numpy as np
import six

from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


def assert_is_semantic_segmentation_dataset(dataset, n_class, repeat=10):
    """Checks if a dataset satisfies semantic segmentation dataset APIs.

    This function checks if a given dataset satisfies semantic segmentation
    dataset dataset APIs or not.
    If the dataset does not satifiy the APIs, this function raises an
    :class:`AssertionError`.

    Args:
        dataset: A dataset to be checked.
        n_class (int): The number of classes including background.
        repeat (int): The number of trials. This function picks
            an example randomly and checks it. This argmuments determines,
            how many times this function picks and checks.
            The default value is :obj:`10`.
    """

    assert len(dataset) > 0, 'The length of dataset must be greater than zero.'

    for _ in six.moves.range(repeat):
        i = np.random.randint(0, len(dataset))
        sample = dataset[i]

        assert len(sample) >= 2, \
            'Each example must have at least two elements:' \
            'img and label.'

        img, label = sample[:2]

        assert_is_image(img, color=True)

        assert isinstance(label, np.ndarray), \
            'label must be a numpy.ndarray.'
        assert label.dtype == np.int32, \
            'The type of label must be numpy.int32.'
        assert label.shape == img.shape[1:], \
            'The shape of label must be (H, W).'
        assert label.min() >= 0 and label.max() < n_class, \
            'The value of label must be in [0, n_class - 1].'
