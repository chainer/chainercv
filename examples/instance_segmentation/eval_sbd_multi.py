import argparse

import chainer
from chainer import iterators
import chainermn

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.datasets import SBDInstanceSegmentationDataset
from chainercv.evaluations import eval_instance_segmentation_voc
from chainercv.experimental.links import FCISResNet101
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('fcis_resnet101',),
        default='fcis_resnet101')
    parser.add_argument('--pretrained-model')
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    args = parser.parse_args()

    if args.model == 'fcis_resnet101':
        if args.pretrained_model:
            model = FCISResNet101(
                n_fg_class=len(sbd_instance_segmentation_label_names),
                pretrained_model=args.pretrained_model)
        else:
            model = FCISResNet101(pretrained_model='sbd')

    model.use_preset('evaluate')

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if not comm.rank == 0:
        apply_to_iterator(model.predict, None, comm=comm)
        return

    dataset = SBDInstanceSegmentationDataset(split='val')
    iterator = iterators.SerialIterator(
        dataset, comm.size, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)), comm=comm)
    # delete unused iterators explicitly
    del in_values

    pred_masks, pred_labels, pred_scores = out_values
    gt_masks, gt_labels = rest_values

    result = eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, args.iou_thresh,
        use_07_metric=True)

    print('')
    print('mAP: {:f}'.format(result['map']))
    for l, name in enumerate(sbd_instance_segmentation_label_names):
        if result['ap'][l]:
            print('{:s}: {:f}'.format(name, result['ap'][l]))
        else:
            print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
