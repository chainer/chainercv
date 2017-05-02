import argparse
import matplotlib.pyplot as plot

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import utils
from chainercv import visualizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(n_classes=20, pretrained_model=args.model)

    img = utils.read_image(args.image, color=True)
    bbox, label, score = model.predict(img, score_threshold=0.6)

    visualizations.vis_bbox(
        img, bbox, label, voc_utils.pascal_voc_labels, score)
    plot.show()


if __name__ == '__main__':
    main()
