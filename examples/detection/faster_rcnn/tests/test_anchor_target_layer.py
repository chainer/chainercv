import chainer
import cv2 as cv
import numpy as np
import six
import unittest

import sys
sys.path.append('..')
from lib.region_proporsal.anchor_target_layer import AnchorTargetLayer  # NOQA


class TestAnchorTargetLayer(unittest.TestCase):

    def setUp(self):
        self.feat_stride = 16
        self.n_channels, self.width, self.height = 16, 14, 14
        self.x = np.arange(1 * self.n_channels *
                           self.height * self.width, dtype=np.float32)
        self.x = self.x.reshape(1, self.n_channels, self.height, self.width)
        img_H, img_W = 224, 224
        self.im_info = np.array([[img_H, img_W, 0.85]])
        self.anchor_target_layer = AnchorTargetLayer(
            self.feat_stride, 2 ** np.arange(1, 6))
        self.height, self.width = self.x.shape[2:]
        self.shifts = self.anchor_target_layer._generate_shifts(
            self.width, self.height)
        self.all_anchors, self.total_anchors = \
            self.anchor_target_layer._generate_proposals(self.shifts)

        self.inds_inside, self.anchors = self.anchor_target_layer._keep_inside(
            self.all_anchors, img_H, img_W)

        self.gt_boxes = np.array([
            [10, 10, 60, 200, 0],
            [50, 100, 210, 210, 1],
            [160, 40, 200, 70, 2]
        ])
        gt_canvas = np.zeros((224, 224))
        for gt in self.gt_boxes:
            cv.rectangle(gt_canvas, (gt[0], gt[1]), (gt[2], gt[3]), 255)
        cv.imwrite('tests/gt_boxes.png', gt_canvas)

        self.argmax_overlaps, self.max_overlaps, self.gt_max_overlaps, \
            self.gt_argmax_overlaps = self.anchor_target_layer._calc_overlaps(
                self.anchors, self.gt_boxes, self.inds_inside)

        self.argmax_overlaps, self.labels = \
            self.anchor_target_layer._create_labels(
                self.inds_inside, self.anchors, self.gt_boxes)

    def test_generate_anchors(self):
        anchor_target_layer = AnchorTargetLayer()
        ret = np.array([[-83.,  -39.,  100.,   56.],
                        [-175.,  -87.,  192.,  104.],
                        [-359., -183.,  376.,  200.],
                        [-55.,  -55.,   72.,   72.],
                        [-119., -119.,  136.,  136.],
                        [-247., -247.,  264.,  264.],
                        [-35.,  -79.,   52.,   96.],
                        [-79., -167.,   96.,  184.],
                        [-167., -343.,  184.,  360.]]) - 1
        self.assertEqual(anchor_target_layer.anchors.shape, ret.shape)
        np.testing.assert_array_equal(anchor_target_layer.anchors, ret)

        ret = self.anchor_target_layer.anchors
        min_x = ret[:, 0].min()
        min_y = ret[:, 1].min()
        max_x = ret[:, 2].max()
        max_y = ret[:, 3].max()
        canvas = np.zeros(
            (int(abs(min_y) + max_y) + 1,
             int(abs(min_x) + max_x) + 1), dtype=np.uint8)
        ret[:, 0] -= min_x
        ret[:, 2] -= min_x
        ret[:, 1] -= min_y
        ret[:, 3] -= min_y
        for anchor in ret:
            anchor = list(six.moves.map(int, anchor))
            cv.rectangle(
                canvas, (anchor[0], anchor[1]), (anchor[2], anchor[3]), 255)
        cv.imwrite('tests/anchors.png', canvas)

    def test_generate_shifts(self):
        for i in range(len(self.shifts)):
            self.assertEqual(self.shifts[i][0], self.shifts[i][2])
            self.assertEqual(self.shifts[i][1], self.shifts[i][3])
        i = 0
        for y in range(self.height):
            for x in range(self.width):
                xx = x * self.feat_stride
                yy = y * self.feat_stride
                self.assertEqual(len(self.shifts[i]), 4)
                self.assertEqual(self.shifts[i][0], xx)
                self.assertEqual(self.shifts[i][1], yy)
                self.assertEqual(self.shifts[i][2], xx)
                self.assertEqual(self.shifts[i][3], yy)
                i += 1
        self.assertEqual(i, len(self.shifts))

        min_x = self.shifts[:, 0].min()
        min_y = self.shifts[:, 1].min()
        max_x = self.shifts[:, 2].max()
        max_y = self.shifts[:, 3].max()
        canvas = np.zeros(
            (int(abs(min_y) + max_y) + 1,
             int(abs(min_x) + max_x) + 1), dtype=np.uint8)
        shifts = self.shifts.copy()
        shifts[:, 0] -= min_x
        shifts[:, 2] -= min_x
        shifts[:, 1] -= min_y
        shifts[:, 3] -= min_y
        for anchor in shifts:
            anchor = list(six.moves.map(int, anchor))
            cv.circle(canvas, (anchor[0], anchor[1]), 1, 255, -1)
        cv.imwrite('tests/shifts.png', canvas)

    def test_generate_proposals(self):
        self.assertEqual(self.total_anchors, len(self.shifts) *
                         self.anchor_target_layer.anchors.shape[0])

        min_x = self.all_anchors[:, 0].min()
        min_y = self.all_anchors[:, 1].min()
        max_x = self.all_anchors[:, 2].max()
        max_y = self.all_anchors[:, 3].max()
        canvas = np.zeros(
            (int(abs(min_y) + max_y) + 1,
             int(abs(min_x) + max_x) + 1), dtype=np.uint8)
        self.all_anchors[:, 0] -= min_x
        self.all_anchors[:, 1] -= min_y
        self.all_anchors[:, 2] -= min_x
        self.all_anchors[:, 3] -= min_y
        for anchor in self.all_anchors:
            anchor = list(six.moves.map(int, anchor))
            cv.rectangle(
                canvas, (anchor[0], anchor[1]), (anchor[2], anchor[3]), 255)
        cv.imwrite('tests/all_anchors.png', canvas)

    def test_keep_inside(self):
        anchors = self.inds_inside

        min_x = anchors[:, 0].min()
        min_y = anchors[:, 1].min()
        max_x = anchors[:, 2].max()
        max_y = anchors[:, 3].max()
        canvas = np.zeros(
            (int(max_y - min_y) + 1,
             int(max_x - min_x) + 1), dtype=np.uint8)
        anchors[:, 0] -= min_x
        anchors[:, 1] -= min_y
        anchors[:, 2] -= min_x
        anchors[:, 3] -= min_y
        for i, anchor in enumerate(anchors):
            anchor = list(six.moves.map(int, anchor))
            _canvas = np.zeros(
                (int(max_y - min_y) + 1,
                 int(max_x - min_x) + 1), dtype=np.uint8)
            cv.rectangle(
                _canvas, (anchor[0], anchor[1]), (anchor[2], anchor[3]), 255)
            cv.rectangle(
                canvas, (anchor[0], anchor[1]), (anchor[2], anchor[3]), 255)
            cv.imwrite('tests/anchors_inside_{}.png'.format(i), _canvas)
        cv.imwrite('tests/anchors_inside.png'.format(i), canvas)

    def test_calc_overlaps(self):
        self.assertEqual(len(self.anchors), len(self.max_overlaps))
        self.assertEqual(len(self.gt_max_overlaps), len(self.gt_boxes))
        self.assertEqual(len(self.gt_argmax_overlaps), len(self.gt_boxes))
        canvas = np.zeros((int(self.im_info[0, 0]), int(self.im_info[0, 1])))
        for bbox in self.anchors[self.gt_argmax_overlaps]:
            x1, y1, x2, y2 = list(map(int, bbox))
            cv.rectangle(canvas, (x1, y1), (x2, y2), 255)
        cv.imwrite('tests/max_overlap_anchors.png', canvas)

    def test_create_labels(self):
        self.assertEqual(len(self.labels), len(self.anchors))
        neg_ids = np.where(self.labels == 0)[0]
        pos_ids = np.where(self.labels == 1)[0]
        canvas = np.zeros((int(self.im_info[0, 0]), int(self.im_info[0, 1])))
        for bbox in self.anchors[pos_ids]:
            x1, y1, x2, y2 = list(map(int, bbox))
            cv.rectangle(canvas, (x1, y1), (x2, y2), 255)
        cv.imwrite('tests/pos_labels.png', canvas)
        np.testing.assert_array_less(
            self.max_overlaps[neg_ids],
            self.anchor_target_layer.RPN_NEGATIVE_OVERLAP)
        # np.testing.assert_array_less(
        #     self.anchor_target_layer.RPN_POSITIVE_OVERLAP,
        #     self.max_overlaps[pos_ids])

    def test_calc_inside_weights(self):
        bbox_inside_weights = \
            self.anchor_target_layer._calc_inside_weights(
                self.inds_inside, self.labels)
        neg_ids = np.where(self.labels == 0)[0]
        pos_ids = np.where(self.labels == 1)[0]
        ignore_ids = np.where(self.labels == -1)[0]
        np.testing.assert_array_equal(bbox_inside_weights[pos_ids], 1.)
        np.testing.assert_array_equal(bbox_inside_weights[neg_ids], 0.)
        np.testing.assert_array_equal(bbox_inside_weights[ignore_ids], 0.)

    def test_calc_outside_weights(self):
        self.anchor_target_layer.RPN_POSITIVE_WEIGHT = -1
        bbox_outside_weights = \
            self.anchor_target_layer._calc_outside_weights(
                self.inds_inside, self.labels)
        neg_ids = np.where(self.labels == 0)[0]
        pos_ids = np.where(self.labels == 1)[0]

        self.assertEqual(len(np.unique(bbox_outside_weights[pos_ids])), 1)
        self.assertEqual(len(np.unique(bbox_outside_weights[neg_ids])), 1)
        self.assertEqual(np.unique(bbox_outside_weights[pos_ids]),
                         np.unique(bbox_outside_weights[neg_ids]))
        np.testing.assert_array_equal(
            bbox_outside_weights[pos_ids], 1. / np.sum(self.labels >= 0))
        np.testing.assert_array_equal(
            bbox_outside_weights[neg_ids], 1. / np.sum(self.labels >= 0))

        self.anchor_target_layer.RPN_POSITIVE_WEIGHT = 0.8
        bbox_outside_weights = \
            self.anchor_target_layer._calc_outside_weights(
                self.inds_inside, self.labels)
        np.testing.assert_array_equal(
            bbox_outside_weights[pos_ids], 0.8 / np.sum(self.labels == 1))
        np.testing.assert_array_equal(
            bbox_outside_weights[neg_ids], 0.2 / np.sum(self.labels == 0))

    def test_mapup_to_anchors(self):
        bbox_inside_weights = \
            self.anchor_target_layer._calc_inside_weights(
                self.inds_inside, self.labels)
        bbox_outside_weights = \
            self.anchor_target_layer._calc_outside_weights(
                self.inds_inside, self.labels)
        bbox_targets = np.zeros((len(self.inds_inside), 4), dtype=np.float32)
        bbox_targets = self.anchor_target_layer._compute_targets(
            self.anchors, self.gt_boxes[self.argmax_overlaps, :])
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self.anchor_target_layer._mapup_to_anchors(
                self.labels, self.total_anchors, self.inds_inside,
                bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self.assertEqual(len(labels), len(self.all_anchors))
        self.assertEqual(len(bbox_targets), len(self.all_anchors))
        self.assertEqual(len(bbox_inside_weights), len(self.all_anchors))
        self.assertEqual(len(bbox_outside_weights), len(self.all_anchors))

    def test_call(self):
        xp = chainer.cuda.cupy
        x = chainer.Variable(xp.asarray(self.x, dtype=xp.float32))
        gt_boxes = self.gt_boxes
        im_info = self.im_info
        H, W = x.shape[2:]
        img_H, img_W = im_info[0, :2]
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self.anchor_target_layer(gt_boxes, (H, W), (img_H, img_W))

        n_anchors = self.anchor_target_layer.n_anchors
        self.assertEqual(labels.shape,
                         (1, n_anchors, self.height, self.width))
        self.assertEqual(bbox_targets.shape,
                         (1, n_anchors * 4, self.height, self.width))
        self.assertEqual(bbox_inside_weights.shape,
                         (1, n_anchors * 4, self.height, self.width))
        self.assertEqual(bbox_outside_weights.shape,
                         (1, n_anchors * 4, self.height, self.width))
