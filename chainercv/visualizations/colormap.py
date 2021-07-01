import numpy as np


def voc_colormap(labels):
    """Color map used in PASCAL VOC

    Args:
        labels (iterable of ints): Class ids.

    Returns:
        numpy.ndarray: Colors in RGB order. The shape is :math:`(N, 3)`,
        where :math:`N` is the size of :obj:`labels`. The range of the values
        is :math:`[0, 255]`.

    """
    colors = []
    for label in labels:
        r, g, b = 0, 0, 0
        i = label
        for j in range(8):
            if i & (1 << 0):
                r |= 1 << (7 - j)
            if i & (1 << 1):
                g |= 1 << (7 - j)
            if i & (1 << 2):
                b |= 1 << (7 - j)
            i >>= 3
        colors.append((r, g, b))
    return np.array(colors, dtype=np.float32)
