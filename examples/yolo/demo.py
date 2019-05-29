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
    parser.add_argument('--pretrained-model')
    parser.add_argument(
        '--dataset', choices=('voc',), default='voc')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'yolo_v2':
        cls = YOLOv2
    elif args.model == 'yolo_v2_tiny':
        cls = YOLOv2Tiny
    elif args.model == 'yolo_v3':
        cls = YOLOv3

    if args.dataset == 'voc':
        if args.pretrained_model is None:
            args.pretrained_model = 'voc0712'
        label_names = voc_bbox_label_name

    model = cls(pretrained_model=args.pretrained_model,
                **cls.preset_params[args.dataset])

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=label_names)
    plt.show()


if __name__ == '__main__':
    main()
