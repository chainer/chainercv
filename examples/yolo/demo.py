import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.experimental.links import YOLOv2Tiny
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('yolo_v2', 'yolo_v2_tiny', 'yolo_v3'),
        default='yolo_v2')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='voc0712')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'yolo_v2':
        model = YOLOv2(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'yolo_v2_tiny':
        model = YOLOv2Tiny(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'yolo_v3':
        model = YOLOv3(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    main()
