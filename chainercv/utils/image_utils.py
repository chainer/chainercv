import numpy as np
from PIL import Image


def read_image_as_array(path, dtype=np.uint8, copy=True, force_color=False):
    f = Image.open(path)
    try:
        if len(f.getbands()) == 1 and force_color:
            img = f.convert('RGB')
        else:
            img = f
        img = np.asarray(img, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    if copy:
        # you need this to make the array editable
        img = img.copy()

    if img.ndim == 2:
        return img[np.newaxis]
    else:
        return img.transpose(2, 0, 1)[::-1]
