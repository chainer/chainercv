from chainer.backends import cuda
import numpy as np


def mask_to_bbox(mask):
    """Compute the bounding boxes around the masked regions.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs.

    Args:
        mask (array): An array whose shape is :math:`(R, H, W)`.
            :math:`R` is the number of masks.
            The dtype should be :obj:`numpy.bool`.

    Returns:
        array:
        The bounding boxes around the masked regions.
        This is an array whose shape is :math:`(R, 4)`.
        :math:`R` is the number of bounding boxes.
        The dtype should be :obj:`numpy.float32`.

    """
    R, H, W = mask.shape
    xp = cuda.get_array_module(mask)

    instance_index, ys, xs = xp.nonzero(mask)
    bbox = xp.zeros((R, 4), dtype=np.float32)
    for i in range(R):
        ys_i = ys[instance_index == i]
        xs_i = xs[instance_index == i]
        if len(ys_i) == 0:
            continue
        y_min = ys_i.min()
        x_min = xs_i.min()
        y_max = ys_i.max() + 1
        x_max = xs_i.max() + 1
        bbox[i] = xp.array([y_min, x_min, y_max, x_max], dtype=np.float32)
    return bbox
