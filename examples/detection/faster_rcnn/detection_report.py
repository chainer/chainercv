import chainer
from chainer.training.extension import PRIORITY_WRITER
from chainer.dataset.convert import concat_examples
from chainer.reporter import report

from chainercv.evaluations import eval_detection


def _identity(x):
    return x


class DetectionReport(chainer.training.Extension):

    priority=PRIORITY_WRITER

    def __init__(self, model, dataset, device, n_class,
                 minoverlap=0.5, use_07_metric=False,
                 post_transform=_identity
                 ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.n_class = n_class
        self.minoverlap = minoverlap
        self.use_07_metric = use_07_metric
        self.post_transform = post_transform

    def __call__(self, trainer):
        cpu = self.model._cpu
        self.model.to_gpu(self.device)
        bboxes, labels, confs, gt_bboxes, gt_labels, gt_difficults = _record_bbox(
            self.model, self.dataset, self.device, self.n_class, self.post_transform)
        metric = eval_detection(
            bboxes, labels, confs, gt_bboxes, gt_labels,
            n_class=self.n_class, gt_difficults=gt_difficults,
            minoverlap=self.minoverlap, use_07_metric=self.use_07_metric)

        print metric['map']
        report({'map': metric['map']})
        if cpu:
            self.model.to_cpu()


def _record_bbox(model, dataset, device, n_class, post_transform=_identity):
    bboxes = []
    confs = []
    labels = []
    gt_bboxes = []
    gt_labels = []
    gt_difficults = []
    for i in range(len(dataset)):
        batch = dataset[i:i+1]
        # ground truth
        _, bbox, label, _, difficult = post_transform(batch[0])
        gt_bboxes.append(bbox)  # (M, 4)
        gt_labels.append(label)  # (M,)
        gt_difficults.append(difficult)  # (M,)

        in_arrays = concat_examples(batch, device)
        in_vars = tuple(chainer.Variable(x, volatile=True) for x in in_arrays)
        # note that bbox is in original scale
        img, bbox, label, scale, _ = in_vars
        bbox = chainer.cuda.to_cpu(bbox.data)[0]  # (M, 4)
        label = chainer.cuda.to_cpu(label.data)[0]  # (M,)

        pred_bbox, pred_label, pred_confidence = model.predict(img, scale=scale)
        pred_bbox = chainer.cuda.to_cpu(pred_bbox)[0]  # (N, 4)
        pred_label = chainer.cuda.to_cpu(pred_label)[0]  # (N,)
        pred_confidence = chainer.cuda.to_cpu(pred_confidence)[0]  # (N,)

        bboxes.append(pred_bbox)
        labels.append(pred_label)
        confs.append(pred_confidence)
    return bboxes, labels, confs, gt_bboxes, gt_labels, gt_difficults
