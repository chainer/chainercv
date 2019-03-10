import argparse

import chainer
from chainer import iterators

from chainercv.datasets import COCOKeypointDataset
from chainercv.evaluations import eval_keypoint_detection_coco
from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

models = {
    # model: (class, dataset -> pretrained_model, default batchsize)
    'mask_rcnn_fpn_resnet50': (MaskRCNNFPNResNet50,
                               {}, 1),
    'mask_rcnn_fpn_resnet101': (MaskRCNNFPNResNet101,
                                {}, 1),
}


def setup(dataset, model_name, pretrained_model, batchsize):
    cls, pretrained_models, default_batchsize = models[model_name]
    dataset_name = dataset
    if pretrained_model is None:
        pretrained_model = pretrained_models.get(dataset_name, dataset_name)
    if batchsize is None:
        batchsize = default_batchsize

    if dataset_name == 'coco':
        dataset = COCOKeypointDataset(
            split='val',
            use_crowded=True, return_crowded=True,
            return_area=True)
        n_fg_class = 1
        n_point = 17
        model = cls(
            n_fg_class=n_fg_class,
            pretrained_model=pretrained_model,
            n_point=n_point,
            mode='keypoint'
        )
        model.use_preset('evaluate')

        def eval_(out_values, rest_values):
            (pred_points, pred_point_scores, pred_labels, pred_scores,
             pred_bboxes) = out_values
            (gt_points, gt_visibles, gt_labels, gt_bboxes,
             gt_areas, gt_crowdeds) = rest_values

            result = eval_keypoint_detection_coco(
                pred_points, pred_labels, pred_scores,
                gt_points, gt_visibles, gt_labels, gt_bboxes,
                gt_areas, gt_crowdeds)

            print()
            for area in ('all', 'large', 'medium'):
                print('mmAP ({}):'.format(area),
                      result['map/iou=0.50:0.95/area={}/max_dets=20'.format(
                          area)])

    return dataset, eval_, model, batchsize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('coco',), default='coco')
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

    iterator = iterators.MultithreadIterator(
        dataset, batchsize, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
