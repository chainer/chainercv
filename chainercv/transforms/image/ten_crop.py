import numpy as np


def ten_crop(img, size):
    """Crop 10 regions from an array.

    This method crops 10 regions. All regions will be in shape
    :obj:`size`. These regions consist of 1 center crop and 4 corner
    crops and horizontal flips of them.

    The crops are ordered in this order.

    * center crop
    * top-left crop
    * bottom-left crop
    * top-right crop
    * bottom-right crop
    * center crop (flipped horizontally)
    * top-left crop (flipped horizontally)
    * bottom-left crop (flipped horizontally)
    * top-right crop (flipped horizontally)
    * bottom-right crop (flipped horizontally)

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        size (tuple): The size of output images after cropping.
            This value is :math:`(height, width)`.

    Returns:
        The cropped arrays. The shape of tensor is :math:`(10, C, H, W)`.

    """
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('shape of image is smaller than output shape')

    crops = np.stack((
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, iH - H:iH, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, iH - H:iH, iW - W:iW],
    ))

    crops = np.vstack((crops, crops[:, :, :, ::-1]))

    return crops
