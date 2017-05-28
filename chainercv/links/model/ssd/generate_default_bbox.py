from __future__ import division

import itertools
import numpy as np


def generate_default_bbox(grids, aspect_ratios, steps, sizes):
    """Generate a set of default bounding boxes.

    This function generates a set of default bounding boxes
    which is used in Single Shot Multibox Detector [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        grids (iterable of ints): An iterable of integers.
            Each integer indicates the size of feature map.
        aspect_ratios (iterable of tuples of ints)`:
            An iterable of tuples of integers.
            Each tuple indicates the aspect ratios of default bounding boxes
            at each feature maps.
            The length of this iterable should be :obj:`len(grids)`.
        steps (iterable of floats): The step size for each feature map.
            The length of this iterable should be :obj:`len(grids)`.
        sizes (iterable of floats): The base size of default bounding boxes
            for each feature map.
            The length of this iterable should be :obj:`len(grids) + 1`.

    Returns:
        ~numpy.ndarray:
        An array whose shape is :math:`(K, 4)`, where :math:`K` is
        the number of default bounding boxes. Each bounding box is
        organized by :obj:`(center_x, center_y, width, height)`.
    """

    if not len(aspect_ratios) == len(grids):
        raise ValueError('The length of aspect_ratios is wrong.')
    if not len(steps) == len(grids):
        raise ValueError('The length of steps is wrong.')
    if not len(sizes) == len(grids) + 1:
        raise ValueError('The length of sizes is wrong.')

    default_bbox = list()

    for k, grid in enumerate(grids):
        for v, u in itertools.product(range(grid), repeat=2):
            cx = (u + 0.5) * steps[k]
            cy = (v + 0.5) * steps[k]

            s = sizes[k]
            default_bbox.append((cx, cy, s, s))

            s = np.sqrt(sizes[k] * sizes[k + 1])
            default_bbox.append((cx, cy, s, s))

            s = sizes[k]
            for ar in aspect_ratios[k]:
                default_bbox.append(
                    (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                default_bbox.append(
                    (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))

    default_bbox = np.stack(default_bbox)
    return default_bbox
