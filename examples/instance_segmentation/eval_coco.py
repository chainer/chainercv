import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.evaluations import eval_instance_segmentation_coco
from chainercv.experimental.links import FCISResNet101
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('fcis_resnet101',),
        default='fcis_resnet101')
    parser.add_argument('--pretrained-model')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    if args.model == 'fcis_resnet101':
        if args.pretrained_model:
            pretrained_model = args.pretrained_model
        else:
            pretrained_model = 'coco'
        proposal_creator_params = FCISResNet101.proposal_creator_params
        proposal_creator_params['min_size'] = 2
        model = FCISResNet101(
            n_fg_class=len(coco_instance_segmentation_label_names),
            anchor_scales=(4, 8, 16, 32),
            pretrained_model=pretrained_model,
            proposal_creator_params=proposal_creator_params)

    model.use_preset('coco_evaluate')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = COCOInstanceSegmentationDataset(
        split='minival', year='2014',
        use_crowded=True, return_crowded=True, return_area=True)
    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    pred_masks, pred_labels, pred_scores = out_values
    gt_masks, gt_labels, gt_areas, gt_crowdeds = rest_values

    result = eval_instance_segmentation_coco(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, gt_areas, gt_crowdeds)

    keys = [
        'map/iou=0.50:0.95/area=all/max_dets=100',
        'map/iou=0.50/area=all/max_dets=100',
        'map/iou=0.75/area=all/max_dets=100',
        'map/iou=0.50:0.95/area=small/max_dets=100',
        'map/iou=0.50:0.95/area=medium/max_dets=100',
        'map/iou=0.50:0.95/area=large/max_dets=100',
        'mar/iou=0.50:0.95/area=all/max_dets=1',
        'mar/iou=0.50:0.95/area=all/max_dets=10',
        'mar/iou=0.50:0.95/area=all/max_dets=100',
        'mar/iou=0.50:0.95/area=small/max_dets=100',
        'mar/iou=0.50:0.95/area=medium/max_dets=100',
        'mar/iou=0.50:0.95/area=large/max_dets=100',
    ]

    print('')
    for key in keys:
        print('{:s}: {:f}'.format(key, result[key]))


if __name__ == '__main__':
    main()
