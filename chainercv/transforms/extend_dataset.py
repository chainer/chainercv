import os
import shelve
import tempfile


def extend(dataset, transform, method_name='get_example'):
    """Extend a method to transform examples after extracted.

    This method updates a method of a dataset to apply a transformation to
    an example after extraction.
    The :obj:`transform` is expected to be a function that takes output of
    a method specified  by :obj:`method_name` and returns the transformation
    of the output.
    The method specified by :obj:`method_name` is expected to take inputs
    :obj:`self` (a dataset) and :obj:`i` (an index to an example). These
    arguments are the input to
    :func:`chainer.dataset.DatasetMixin.get_example`, which is a canonical
    example of the method that is decorated by :func:`extend`.

    By convention of Chainer, the image arrays are in CHW format. The
    output of :obj:`transform` is expected to follow this convention.

    Args:
        dataset (~chainer.dataset.DatasetMixin): a dataset whose method
            will be decorated.
        transform (function): takes an example retrieved by :obj:`dataset`
            and returns the transformed.
        method_name (string): name of the :obj:`dataset`'s method which
            will be decorated.

    .. note::

        The :obj:`transform` is typically a user defined function. Here is
        an example of the usage from a Semantic Segmentation task.
        In this example, both a color image and a label image need to be
        padded to match the given shape. In addition to the padding operation,
        color image needs to be subtracted by a constant.

        >>> from chainercv.datasets import VOCSemanticSegmentationDataset
        >>> from chainercv.transforms import extend
        >>> from chainercv.transforms import pad
        >>> dataset = VOCSemanticSegmentationDataset()
        >>> def transform(in_data):
        >>>     img, label = in_data
        >>>     img = pad(img, (512, 512), bg_value=0)
        >>>     label = pad(label, (512, 512), bg_value=-1)
        >>>     img -= 122.5
        >>>     return img, label
        >>> extend(dataset, transform)
        >>> img, label = dataset.get_example(0)

    """
    method = getattr(dataset, method_name)

    def _extended(i):
        in_data = method(i)
        return transform(in_data)
    setattr(dataset, method_name, _extended)


def extend_cache(dataset, transform, method_name='get_example'):
    """Extend a method to transform examples and cache the result.

    This method updates a method of a dataset to apply a transformation to
    an example after extraction and cache the result.
    This method behaves similar to :func:`chainer.transform.extend` except
    that this caches the result of the transformed data.

    The cached data is organized by indexing keys which are integers that
    are used to access examples of :class:`chainer.dataset.DatasetMixin`.
    If the query index has been previously indexed before, the cached data
    will be retrieved.
    Note that :obj:`transform` needs to be a determinisitic function. If not,
    due to the nature of caching, the non-deterministic property of the
    function will be lost. Namely, the example extract for the first time for
    the given index will be retrieved from the second time.

    Args:
        dataset (~chainer.dataset.DatasetMixin): a dataset whose method
            will be decorated.
        transform (function): takes an example retrieved by :obj:`dataset`
            and returns the transformed.
        method_name (string): name of the :obj:`dataset`'s method which
            will be decorated.

    .. seealso::
        :func:`chainercv.transform.extend`.

    """
    filename = os.path.join(tempfile.mkdtemp(), 'chainercv.db')
    cache = shelve.open(filename, protocol=2)

    method = getattr(dataset, method_name)

    def _extended(i):
        key = str(i)
        if key not in cache:
            in_data = method(i)
            cache[key] = transform(in_data)
        return cache[key]
    setattr(dataset, method_name, _extended)
