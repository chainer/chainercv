from __future__ import division


from chainer.backends import cuda


def mask_iou(mask_a, mask_b):
    """Calculate the Intersection of Unions (IoUs) between masks.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`mask_a` and :obj:`mask_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        mask_a (array): An array whose shape is :math:`(N, H, W)`.
            :math:`N` is the number of masks.
            The dtype should be :obj:`numpy.bool`.
        mask_b (array): An array similar to :obj:`mask_a`,
            whose shape is :math:`(K, H, W)`.
            The dtype should be :obj:`numpy.bool`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th mask in :obj:`mask_a` and :math:`k` th mask \
        in :obj:`mask_b`.

    """
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    xp = cuda.get_array_module(mask_a)

    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = xp.empty((n_mask_a, n_mask_b), dtype=xp.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = xp.bitwise_and(m_a, m_b).sum()
            union = xp.bitwise_or(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou
