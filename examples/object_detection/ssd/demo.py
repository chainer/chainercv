import matplotlib.pyplot as plot
import numpy as np
from skimage.data import astronaut

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import tasks
from chainercv import transforms


def main():
    # load pre-trained model
    ssd = SSD300()

    # convert image to CHW, BGR and float32
    img = astronaut().transpose(2, 0, 1)[::-1].astype(np.float32)
    bbox, label, _ = ssd.predict(img, min_conf=0.6)

    vis_img = transforms.chw_to_pil_image(img)
    tasks.vis_bbox(vis_img, bbox, label, voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
