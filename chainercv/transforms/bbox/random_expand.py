import numpy
import random


def random_expand(img, bbox, max_ratio=4, fill=0):
    """Expand image randomly.

    This method expands the size of image randomly by padding pixels. The
    aspect ratio of the image is kept. The bounding boxes are move according
    to the expansion.

    This method is used in training of SSD [1].

    .. [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, \
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg. \
    SSD: Single Shot MultiBox Detector. ECCV 2016.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW format.
        bbox (~numpy.ndarray): Bounding boxes to be augmented. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        max_ratio (float): The maximum ratio of expansion. In the original
            paper, this value is 4.
        fill (float or tuple or ~numpy.ndarray): The value of padded pixels.
            In the original paper, this value is the mean of ImageNet.

    Returns:
        An expanded image and the bounding boxes corresponding to the expanded
        image.

    """

    if max_ratio <= 1:
        return img, bbox

    C, H, W = img.shape

    ratio = random.uniform(1, max_ratio)
    out_H, out_W = int(H * ratio), int(W * ratio)

    top = random.randint(0, out_H - H)
    left = random.randint(0, out_W - W)

    out_img = numpy.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = numpy.array(fill).reshape(-1, 1, 1)
    out_img[:, top:top + H, left:left + W] = img

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (left, top)
    out_bbox[:, 2:] += (left, top)

    return out_img, out_bbox
