import matplotlib.pyplot as plot
import numpy as np
from skimage.data import astronaut

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import tasks
from chainercv import transforms


def main():
    ssd = SSD300()

    img = astronaut().transpose(2, 0, 1)[::-1].astype(np.float32)
    bbox, conf = ssd.predict(img)

    conf, label = conf[:, 1:].max(axis=1), conf.argmax(axis=1)
    mask = conf > 0.6
    bbox, conf, label = bbox[mask], conf[mask], label[mask]
    order = conf.argsort()[::-1]
    bbox, label = bbox[order], label[order]

    bbox, param = transforms.non_maximum_suppression(
        bbox, 0.45, return_param=True)
    label = label[param['selection']]

    vis_img = transforms.chw_to_pil_image(img)
    tasks.vis_bbox(vis_img, bbox, label, voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
