import random


def random_crop(xs, output_shape, return_slices=False):
    """Crop array randomly into `output_shape`.

    All arrays will be cropped by the same region randomly selected. The
    output will all be in shape ``output_shape``.

    Args:
        xs (tuple or list of arrays or an numpy.ndarray): These arrays shoule
            have same shape if this is list or tuple.
        output_shape (tuple): Shape of the array after cropping. If ``None``
            is included in the tuple, that dimension will not be cropped.

    Returns:
        If the input ``xs`` is tuple or list,
        this is a tuple. If ``xs`` is an numpy.ndarray, numpy.ndarray
        will be returned.

    """
    force_array = False
    if not isinstance(xs, tuple):
        xs = (xs,)
        force_array = True
    if len(output_shape) != xs[0].ndim:
        raise ValueError

    x = xs[0]
    slices = []
    for i, dim in enumerate(output_shape):
        if dim is None:
            slices.append(slice(None))
            continue
        if x.shape[i] == dim:
            start = 0
        elif x.shape[i] > dim:
            start = random.choice(range(x.shape[i] - dim))
        else:
            raise ValueError('shape of image is larger than output_shape')
        slices.append(slice(start, start + dim))
    slices = tuple(slices)

    outs = []
    for x in xs:
        outs.append(x[slices])

    if force_array:
        outs = outs[0]
    else:
        outs = tuple(outs)

    if return_slices:
        return outs, slices
    else:
        return outs


if __name__ == '__main__':
    from chainercv.datasets import VOCSemanticSegmentationDataset
    dataset = VOCSemanticSegmentationDataset()
    img, label = dataset.get_example(0)
    img, label = random_crop((img, label), (None, 256, 256))
