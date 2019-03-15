import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import coco_keypoint_names
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.links import KeypointRCNNFPNResNet101
from chainercv.links import KeypointRCNNFPNResNet50
from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainercv.visualizations import vis_instance_segmentation
from chainercv.visualizations import vis_keypoint_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('faster_rcnn_fpn_resnet50', 'faster_rcnn_fpn_resnet101',
                 'mask_rcnn_fpn_resnet50', 'mask_rcnn_fpn_resnet101',
                 'keypoint_rcnn_fpn_resnet50', 'keypoint_rcnn_fpn_resnet101'),
        default='faster_rcnn_fpn_resnet50')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='coco')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'faster_rcnn_fpn_resnet50':
        mode = 'bbox'
        model = FasterRCNNFPNResNet50(
            n_fg_class=len(coco_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'faster_rcnn_fpn_resnet101':
        mode = 'bbox'
        model = FasterRCNNFPNResNet101(
            n_fg_class=len(coco_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'mask_rcnn_fpn_resnet50':
        mode = 'instance_segmentation'
        model = MaskRCNNFPNResNet50(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'mask_rcnn_fpn_resnet101':
        mode = 'instance_segmentation'
        model = MaskRCNNFPNResNet101(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'keypoint_rcnn_fpn_resnet50':
        mode = 'keypoint'
        model = KeypointRCNNFPNResNet50(
            n_fg_class=1,
            pretrained_model=args.pretrained_model,
            n_point=len(coco_keypoint_names[0]))
    elif args.model == 'keypoint_rcnn_fpn_resnet101':
        mode = 'keypoint'
        model = KeypointRCNNFPNResNet101(
            n_fg_class=1,
            pretrained_model=args.pretrained_model,
            n_point=len(coco_keypoint_names[0]))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)

    if mode == 'bbox':
        bboxes, labels, scores = model.predict([img])
        bbox = bboxes[0]
        label = labels[0]
        score = scores[0]

        vis_bbox(
            img, bbox, label, score, label_names=coco_bbox_label_names)
    elif mode == 'instance_segmentation':
        masks, labels, scores = model.predict([img])
        mask = masks[0]
        label = labels[0]
        score = scores[0]
        vis_instance_segmentation(
            img, mask, label, score,
            label_names=coco_instance_segmentation_label_names)
    elif mode == 'keypoint':
        points, labels, scores, point_scores, bboxes = model.predict([img])
        point = points[0]
        label = labels[0]
        score = scores[0]
        point_score = point_scores[0]
        bbox = bboxes[0]
        ax = vis_keypoint_coco(
            img, point, None, point_score)
        vis_bbox(None, bbox, label, score=score,
                 label_names=coco_bbox_label_names, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
