import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('faster_rcnn_fpn_resnet50', 'faster_rcnn_fpn_resnet101'),
        default='faster_rcnn_fpn_resnet50')
    parser.add_argument('--pretrained-model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=1)
    args = parser.parse_args()

    if args.model == 'faster_rcnn_fpn_resnet50':
        if args.pretrained_model:
            model = FasterRCNNFPNResNet50(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model=args.pretrained_model)
        else:
            model = FasterRCNNFPNResNet50(pretrained_model='coco')
    elif args.model == 'faster_rcnn_fpn_resnet101':
        if args.pretrained_model:
            model = FasterRCNNFPNResNet101(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model=args.pretrained_model)
        else:
            model = FasterRCNNFPNResNet101(pretrained_model='coco')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = COCOBboxDataset(
        year='2017',
        split='val',
        use_crowded=True,
        return_area=True,
        return_crowded=True)
    iterator = iterators.MultithreadIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_area, gt_crowded = rest_values

    result = eval_detection_coco(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_area, gt_crowded)

    print()
    for area in ('all', 'large', 'medium', 'small'):
        print('mmAP ({}):'.format(area),
              result['map/iou=0.50:0.95/area={}/max_dets=100'.format(area)])


if __name__ == '__main__':
    main()
