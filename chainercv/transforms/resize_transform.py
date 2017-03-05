import numpy as np
import skimage.transform


def resize(x, output_shape):
    x = x.transpose(1, 2, 0)
    scale = np.max(np.abs(x))
    x = skimage.transform.resize(
        x / scale, output_shape).astype(x.dtype)
    x = x.transpose(2, 0, 1) * scale
    return x
