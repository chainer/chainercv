from __future__ import division

import math
import numpy as np


def convolution_crop(img, size, stride, return_param=False):
    """Strided cropping.

    This extracts cropped images from the input. The cropped images are
    extracted from the entire image, while taking a constant steps between
    neighboring patches.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        size (tuple): The size of output image after cropping.
            This value is :math:`(height, width)`.
        stride (tuple): The stride between crops. This contains
            two values: stride in the vertical and horizontal directions.
        return_param (bool): If :obj:`True`, this function returns
            information of slices.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`crop_imgs` that is a stack of cropped images.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`crop_imgs, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slices** (*list slices*): Slices used to crop the input image.\
            The relation below holds together with :obj:`x_slices`.
        * **x_slices** (*list of slices*): Similar to :obj:`y_slices`.
        * **crop_y_slices** (*list of slices*): This indicates the region of \
            the cropped image that is actually extracted from the input. \
            This is relevant only when borders of the input are cropped.
        * **crop_x_slices** (*list of slices*): Similar to \
            :obj:`crop_y_slices`.

            .. code::

                crop_img = crop_imgs[i][:, crop_y_slices[i], crop_x_slices[i]]
                crop_img == img[:, y_slices[i], x_slices[i]]

    Examples:

        >>> import numpy as np
        >>> from chainercv.datasets import VOCBboxDataset
        >>> from chainercv.transforms import resize
        >>> from chainercv.experimental.links.model.pspnet import \
        ...     convolution_crop
        >>>
        >>> img, _, _ = VOCBboxDataset(year='2007')[0]
        >>> img = resize(img, (300, 300))
        >>> imgs, param = convolution_crop(
        >>>     img, (128, 128), (96, 96), return_param=True)
        >>> # Restore the original image from the cropped images.
        >>> output = np.zeros((3, 300, 300))
        >>> count = np.zeros((300, 300))
        >>> for i in range(len(imgs)):
        >>>     crop_y_slice = param['crop_y_slices'][i]
        >>>     crop_x_slice = param['crop_x_slices'][i]
        >>>     y_slice = param['y_slices'][i]
        >>>     x_slice = param['x_slices'][i]
        >>>     output[:, y_slice, x_slice] +=\
        ...         imgs[i][:, crop_y_slice, crop_x_slice]
        >>>     count[y_slice, x_slice] += 1
        >>> output = output / count[None]
        >>> np.testing.assert_equal(output, img)
        >>>
        >>> # Visualization of the cropped images
        >>> import matplotlib.pyplot as plt
        >>> from chainercv.utils import tile_images
        >>> from chainercv.visualizations import vis_image
        >>> v_imgs = tile_images(imgs, 5, fill=122.5)
        >>> vis_image(v_imgs)
        >>> plt.show()

    """
    _, H, W = img.shape

    h = int(math.ceil((H - size[0]) / stride[0])) + 1
    w = int(math.ceil((W - size[1]) / stride[1])) + 1

    start_y = -(size[0] + stride[0] * (h - 1) - H) // 2
    start_x = -(size[1] + stride[1] * (w - 1) - W) // 2

    crop_imgs = []
    y_slices = []
    x_slices = []
    crop_y_slices = []
    crop_x_slices = []
    for y in range(h):
        for x in range(w):
            y_min = y * stride[0] + start_y
            x_min = x * stride[1] + start_x
            y_max = y_min + size[0]
            x_max = x_min + size[1]

            crop_y_min = np.abs(np.minimum(y_min, 0))
            crop_x_min = np.abs(np.minimum(x_min, 0))
            crop_y_max = size[0] - np.maximum(y_max - H, 0)
            crop_x_max = size[1] - np.maximum(x_max - W, 0)

            crop_img = np.zeros((img.shape[0], size[0], size[1]),
                                dtype=img.dtype)
            y_slice = slice(max(y_min, 0), min(y_max, H))
            x_slice = slice(max(x_min, 0), min(x_max, W))
            crop_y_slice = slice(crop_y_min, crop_y_max)
            crop_x_slice = slice(crop_x_min, crop_x_max)
            crop_img[:, crop_y_slice, crop_x_slice] = img[:, y_slice, x_slice]

            crop_imgs.append(crop_img)
            y_slices.append(y_slice)
            x_slices.append(x_slice)
            crop_y_slices.append(crop_y_slice)
            crop_x_slices.append(crop_x_slice)

    if return_param:
        param = {'y_slices': y_slices, 'x_slices': x_slices,
                 'crop_y_slices': crop_y_slices,
                 'crop_x_slices': crop_x_slices}
        return np.array(crop_imgs), param
    else:
        return np.array(crop_imgs)
