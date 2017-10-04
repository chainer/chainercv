import argparse
import json
import os

import chainer
import chainer.functions as F
from chainercv import transforms
import numpy as np
import pspnet
from skimage import io

from datasets import ADE20KSemanticSegmentationDataset  # NOQA  # isort:skip
from datasets import ADE20KTestImageDataset  # NOQA  # isort:skip
from datasets import CityscapesSemanticSegmentationDataset  # NOQA  # isort:skip
from datasets import CityscapesTestImageDataset  # NOQA  # isort:skip
from datasets import VOCSemanticSegmentationDataset  # NOQA  # isort:skip
from datasets import cityscapes_labels  # NOQA  # isort:skip
from datasets import ade20k_label_colors  # NOQA  # isort:skip


def inference(model, n_class, img, scales):
    h, w = img.shape[1:]
    preds = []
    orig_img = img.copy()
    if scales is not None and isinstance(scales, (list, tuple)):
        for scale in scales:
            if scale != 1.0:
                print('orig_shape:', orig_img.shape, end=' ')
                img = transforms.resize(
                    orig_img, (int(h * scale), int(w * scale)))
                print('scale:', scale, img.shape, end=' ')
            y = model.predict([img], argmax=False)[0]
            if y.shape[1:] != (h, w):
                y = F.resize_images(y[None, ...], (h, w)).data[0]
            else:
                y = y.data[0]
            print('resized img:', y.shape)
            preds.append(y)
        pred = np.mean(preds, axis=0)
    else:
        pred = model.predict([img], argmax=False)[0]
    pred = np.argmax(pred, axis=0)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument(
        '--dataset', type=str, choices=['voc2012', 'cityscapes', 'ade20k'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--start_i', type=int)
    parser.add_argument('--end_i', type=int)
    parser.add_argument('--split', type=str)
    parser.add_argument('--param_fn', type=str)
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
    mean = np.array([123.68, 116.779, 103.939])
    comm = None

    print('command args:')
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.dataset == 'voc2012':
        n_class = 21
        input_size = (473, 473)
        n_blocks = [3, 4, 23, 3]
        feat_size = 60
        mid_stride = True
        pyramids = [6, 3, 2, 1]
        pretrained_model = args.param_fn
        dataset = VOCSemanticSegmentationDataset(
            args.data_dir, split=args.split)
    elif args.dataset == 'cityscapes':
        n_class = 19
        input_size = (713, 713)
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        pyramids = [6, 3, 2, 1]
        pretrained_model = args.param_fn
        if args.split == 'test':
            dataset = CityscapesTestImageDataset(args.data_dir)
        else:
            dataset = CityscapesSemanticSegmentationDataset(
                args.data_dir, 'fine', args.split)
    elif args.dataset == 'ade20k':
        n_class = 150
        input_size = (473, 473)
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        pyramids = [6, 3, 2, 1]
        pretrained_model = args.param_fn
        if args.split == 'test':
            dataset = ADE20KTestImageDataset(args.data_dir)
        else:
            dataset = ADE20KSemanticSegmentationDataset(
                args.data_dir, split=args.split)

    print('{} dataset:'.format(args.dataset), len(dataset))
    assert len(dataset) > 0

    chainer.config.train = False
    model = pspnet.PSPNet(n_class, input_size, n_blocks, pyramids, mid_stride,
                          mean, comm, pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu)
        model.to_gpu(args.gpu)

    for i in range(args.start_i, args.end_i):
        if i >= len(dataset):
            continue
        img = dataset[i]
        if isinstance(img, tuple) and len(img) == 2:
            img = img[0]
        out_fn = os.path.join(
            args.out_dir, os.path.basename(dataset.img_paths[i]))
        pred = inference(model, n_class, img, scales)
        assert pred.ndim == 2

        if args.dataset == 'cityscapes':
            color_out = np.zeros(
                (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            label_out = np.zeros_like(pred)
            for label in cityscapes_labels:
                label_out[np.where(pred == label.trainId)] = label.id
                color_out[np.where(pred == label.trainId)] = label.color
            pred = label_out
            color_fn = out_fn.replace('.', '_color.')
            io.imsave(color_fn, color_out)
        if args.dataset == 'ade20k':
            color_out = np.zeros(
                (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for i, color in enumerate(ade20k_label_colors):
                color_out[np.where(pred == i)] = color
            color_fn = out_fn.replace('.', '_color.')
            io.imsave(color_fn, color_out)
        io.imsave(out_fn, pred)
        print(i, pred.shape, out_fn)
