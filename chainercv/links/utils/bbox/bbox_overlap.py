from chainer import cuda


def bbox_overlap(bbox_a, bbox_b):
    """Calculate Jaccard overlap between bounding boxes.

    Jaccard overlap is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs.
    The output is same type as the type of the inputs.

    Args:
        bbox (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        query_bbox (array): An array similar to :obj:`bbox`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains overlap between \
        :math:`n` th bounding box in :obj:`bbox` and :math:`k` th bounding \
        box in :obj:`query_bbox`.

    """
    if bbox.shape[1] != 4 or query_bbox.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(bbox)

    # left top
    lt = xp.maximum(bbox[:, None, :2], query_bbox[:, :2])
    # right bottom
    rb = xp.minimum(bbox[:, None, 2:], query_bbox[:, 2:])

    area_i = xp.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_b = xp.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
    area_q = xp.prod(query_bbox[:, 2:] - query_bbox[:, :2], axis=1)
    return area_i / (area_b[:, None] + area_q - area_i)
