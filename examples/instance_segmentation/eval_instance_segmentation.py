import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.datasets import SBDInstanceSegmentationDataset
from chainercv.evaluations import eval_instance_segmentation_coco
from chainercv.evaluations import eval_instance_segmentation_voc
from chainercv.experimental.links import FCISResNet101
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

models = {
    # model: (class, dataset -> pretrained_model, default batchsize)
    'fcis_resnet101': (FCISResNet101, {'sbd': 'sbd', 'coco': 'coco'}, 1),
}


def setup(dataset, model_name, pretrained_model, batchsize):
    cls, pretrained_models, default_batchsize = models[model_name]
    dataset_name = dataset
    if pretrained_model is None:
        pretrained_model = pretrained_models.get(dataset_name, dataset_name)
    if batchsize is None:
        batchsize = default_batchsize

    if dataset_name == 'sbd':
        dataset = SBDInstanceSegmentationDataset(split='val')
        label_names = sbd_instance_segmentation_label_names

        param = cls.preset_param(dataset_name)
        param['pretrained_model'] = pretrained_model
        model = cls(**param)
        model.use_preset('evaluate')

        def eval_(out_values, rest_values):
            pred_masks, pred_labels, pred_scores = out_values
            gt_masks, gt_labels = rest_values

            result = eval_instance_segmentation_voc(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels, use_07_metric=True)

            print('')
            print('mAP: {:f}'.format(result['map']))
            for l, name in enumerate(sbd_instance_segmentation_label_names):
                if result['ap'][l]:
                    print('{:s}: {:f}'.format(name, result['ap'][l]))
                else:
                    print('{:s}: -'.format(name))

    elif dataset_name == 'coco':
        dataset = COCOInstanceSegmentationDataset(
            split='minival', year='2014',
            use_crowded=True, return_crowded=True, return_area=True)
        label_names = coco_instance_segmentation_label_names

        param = cls.preset_param(dataset_name)
        param['pretrained_model'] = pretrained_model
        model = cls(**param)
        if model_name == 'fcis_resnet101':
            model.use_preset('coco_evaluate')
        else:
            model.use_preset('evaluate')

        def eval_(out_values, rest_values):
            pred_masks, pred_labels, pred_scores = out_values
            gt_masks, gt_labels, gt_areas, gt_crowdeds = rest_values

            result = eval_instance_segmentation_coco(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels, gt_areas, gt_crowdeds)

            print()
            for area in ('all', 'large', 'medium', 'small'):
                print('mmAP ({}):'.format(area),
                      result['map/iou=0.50:0.95/area={}/max_dets=100'.format(
                          area)])

    return dataset, label_names, eval_, model, batchsize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('sbd', 'coco'))
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    dataset, label_names, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    iterator = iterators.MultithreadIterator(
        dataset, batchsize, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
