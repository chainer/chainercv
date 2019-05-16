import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.evaluations import eval_detection_voc
from chainercv.experimental.links import YOLOv2Tiny
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

models = {
    # model: (class, dataset -> pretrained_model, default batchsize)
    'faster_rcnn': (FasterRCNNVGG16, {'voc': 'voc07'}, 32),
    'faster_rcnn_fpn_resnet50': (FasterRCNNFPNResNet50, {}, 1),
    'faster_rcnn_fpn_resnet101': (FasterRCNNFPNResNet101, {}, 1),
    'ssd300': (SSD300, {'voc': 'voc0712'}, 32),
    'ssd512': (SSD512, {'voc': 'voc0712'}, 16),
    'yolo_v2': (YOLOv2, {'voc': 'voc0712'}, 32),
    'yolo_v2_tiny': (YOLOv2Tiny, {'voc': 'voc0712'}, 32),
    'yolo_v3': (YOLOv3, {'voc': 'voc0712'}, 16),
}


def setup(dataset, model, pretrained_model, batchsize):
    dataset_name = dataset
    if dataset_name == 'voc':
        dataset = VOCBboxDataset(
            year='2007', split='test',
            use_difficult=True, return_difficult=True)
        label_names = voc_bbox_label_names

        def eval_(out_values, rest_values):
            pred_bboxes, pred_labels, pred_scores = out_values
            gt_bboxes, gt_labels, gt_difficults = rest_values

            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults,
                use_07_metric=True)

            print()
            print('mAP: {:f}'.format(result['map']))
            for l, name in enumerate(voc_bbox_label_names):
                if result['ap'][l]:
                    print('{:s}: {:f}'.format(name, result['ap'][l]))
                else:
                    print('{:s}: -'.format(name))

    elif dataset_name == 'coco':
        dataset = COCOBboxDataset(
            year='2017', split='val',
            use_crowded=True, return_area=True, return_crowded=True)
        label_names = coco_bbox_label_names

        def eval_(out_values, rest_values):
            pred_bboxes, pred_labels, pred_scores = out_values
            gt_bboxes, gt_labels, gt_area, gt_crowded = rest_values

            result = eval_detection_coco(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_area, gt_crowded)

            print()
            for area in ('all', 'large', 'medium', 'small'):
                print('mmAP ({}):'.format(area),
                      result['map/iou=0.50:0.95/area={}/max_dets=100'.format(
                          area)])

    cls, pretrained_models, default_batchsize = models[model]
    if pretrained_model is None:
        pretrained_model = pretrained_models.get(dataset_name, dataset_name)
    params = cls.preset_params[dataset_name].copy()
    params['n_fg_class'] = len(label_names)
    model = cls(pretrained_model=pretrained_model, **params)

    if batchsize is None:
        batchsize = default_batchsize

    return dataset, eval_, model, batchsize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('voc', 'coco'))
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    dataset, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    iterator = iterators.MultithreadIterator(
        dataset, batchsize, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
