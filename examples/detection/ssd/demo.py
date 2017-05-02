import argparse
import matplotlib.pyplot as plot

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import visualizations
from chainercv import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(n_classes=20, pretrained_model=args.model)

    img = utils.read_image(args.image, color=True)
    bbox, label, _ = model.predict(img, conf_threshold=0.6)

    visualizations.vis_bbox(img, bbox, label, voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
