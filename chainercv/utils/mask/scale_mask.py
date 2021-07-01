from __future__ import division

import numpy as np
import PIL.Image

import chainer
from chainercv import transforms


def scale_mask(mask, bbox, size):
    """Scale instance segmentation mask while keeping the aspect ratio.

    This function exploits the sparsity of :obj:`mask` to speed up
    resize operation.

    The input image will be resized so that
    the shorter edge will be scaled to length :obj:`size` after
    resizing.

    Args:
        mask (array): An array whose shape is :math:`(R, H, W)`.
            :math:`R` is the number of masks.
            The dtype should be :obj:`numpy.bool`.
        bbox (array): The bounding boxes around the masked region
            of :obj:`mask`. This is expected to be the value
            obtained by :obj:`bbox = chainercv.utils.mask_to_bbox(mask)`.
        size (int): The length of the smaller edge.

    Returns:
        array:
        An array whose shape is :math:`(R, H, W)`.
        :math:`R` is the number of masks.
        The dtype should be :obj:`numpy.bool`.

    """
    xp = chainer.backends.cuda.get_array_module(mask)
    mask = chainer.cuda.to_cpu(mask)
    bbox = chainer.cuda.to_cpu(bbox)

    R, H, W = mask.shape
    if H < W:
        out_size = (size, int(size * W / H))
        scale = size / H
    else:
        out_size = (int(size * H / W), size)
        scale = size / W

    bbox[:, :2] = np.floor(bbox[:, :2])
    bbox[:, 2:] = np.ceil(bbox[:, 2:])
    bbox = bbox.astype(np.int32)
    scaled_bbox = bbox * scale
    scaled_bbox[:, :2] = np.floor(scaled_bbox[:, :2])
    scaled_bbox[:, 2:] = np.ceil(scaled_bbox[:, 2:])
    scaled_bbox = scaled_bbox.astype(np.int32)

    out_mask = xp.zeros((R,) + out_size, dtype=np.bool)
    for i, (m, bb, scaled_bb) in enumerate(
            zip(mask, bbox, scaled_bbox)):
        cropped_m = m[bb[0]:bb[2], bb[1]:bb[3]]
        h = scaled_bb[2] - scaled_bb[0]
        w = scaled_bb[3] - scaled_bb[1]
        cropped_m = transforms.resize(
            cropped_m[None].astype(np.float32),
            (h, w),
            interpolation=PIL.Image.NEAREST)[0]
        if xp != np:
            cropped_m = xp.array(cropped_m)
        out_mask[i, scaled_bb[0]:scaled_bb[2],
                 scaled_bb[1]:scaled_bb[3]] = cropped_m
    return out_mask
