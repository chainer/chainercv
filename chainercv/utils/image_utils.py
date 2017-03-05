import numpy as np
from PIL import Image


def read_image_as_array(path, dtype=np.uint8, copy=True):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    if copy:
        # you need this to make the array editable
        image = image.copy()
    return image


def gray2rgb(img):
    assert img.ndim == 2
    img = Image.fromarray(img)
    img = img.convert('RGB')
    return np.asarray(img)
