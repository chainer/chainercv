from __future__ import division

import numpy as np

from chainercv.datasets import coco_keypoint_names
from chainercv.visualizations.vis_image import vis_image


human_id = 0

coco_point_skeleton = [
    [coco_keypoint_names[human_id].index('left_eye'),
     coco_keypoint_names[human_id].index('right_eye')],
    [coco_keypoint_names[human_id].index('left_eye'),
     coco_keypoint_names[human_id].index('nose')],
    [coco_keypoint_names[human_id].index('right_eye'),
     coco_keypoint_names[human_id].index('nose')],
    [coco_keypoint_names[human_id].index('right_eye'),
     coco_keypoint_names[human_id].index('right_ear')],
    [coco_keypoint_names[human_id].index('left_eye'),
     coco_keypoint_names[human_id].index('left_ear')],
    [coco_keypoint_names[human_id].index('right_shoulder'),
     coco_keypoint_names[human_id].index('right_elbow')],
    [coco_keypoint_names[human_id].index('right_elbow'),
     coco_keypoint_names[human_id].index('right_wrist')],
    [coco_keypoint_names[human_id].index('left_shoulder'),
     coco_keypoint_names[human_id].index('left_elbow')],
    [coco_keypoint_names[human_id].index('left_elbow'),
     coco_keypoint_names[human_id].index('left_wrist')],
    [coco_keypoint_names[human_id].index('right_hip'),
     coco_keypoint_names[human_id].index('right_knee')],
    [coco_keypoint_names[human_id].index('right_knee'),
     coco_keypoint_names[human_id].index('right_ankle')],
    [coco_keypoint_names[human_id].index('left_hip'),
     coco_keypoint_names[human_id].index('left_knee')],
    [coco_keypoint_names[human_id].index('left_knee'),
     coco_keypoint_names[human_id].index('left_ankle')],
    [coco_keypoint_names[human_id].index('right_shoulder'),
     coco_keypoint_names[human_id].index('left_shoulder')],
    [coco_keypoint_names[human_id].index('right_hip'),
     coco_keypoint_names[human_id].index('left_hip')]
]


def vis_keypoint_coco(
        img, point, valid=None,
        point_score=None, thresh=2,
        markersize=3, linewidth=1, ax=None):
    """Visualize bounding boxes inside image.

    """
    if valid.dtype != np.bool:
        raise ValueError('The dtype of `valid` should be np.bool')

    from matplotlib import pyplot as plt

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(coco_point_skeleton) + 2)]

    if point_score is None:
        point_score = np.inf * np.ones(point.shape[:2], dtype=np.float32)

    if valid is not None:
        for i, vld in enumerate(valid):
            point_score[i, np.logical_not(vld)] = -np.inf

    for pnt, pnt_sc in zip(point, point_score):
        for l in range(len(coco_point_skeleton)):
            i0 = coco_point_skeleton[l][0]
            i1 = coco_point_skeleton[l][1]
            s0 = pnt_sc[i0]
            y0 = pnt[i0, 0]
            x0 = pnt[i0, 1]
            s1 = pnt_sc[i1]
            y1 = pnt[i1, 0]
            x1 = pnt[i1, 1]
            if s0 > thresh and s1 > thresh:
                line = ax.plot([x0, x1], [y0, y1])
                plt.setp(line, color=colors[l],
                         linewidth=linewidth, alpha=0.7)
            if s0 > thresh:
                ax.plot(
                    x0, y0, '.', color=colors[l],
                    markersize=markersize, alpha=0.7)
            if s1 > thresh:
                ax.plot(
                    x1, y1, '.', color=colors[l],
                    markersize=markersize, alpha=0.7)

        # for better visualization, add mid shoulder / mid hip
        mid_shoulder = (
            pnt[coco_keypoint_names[human_id].index('right_shoulder'), :2] +
            pnt[coco_keypoint_names[human_id].index('left_shoulder'), :2]) / 2
        mid_shoulder_sc = np.minimum(
            pnt_sc[coco_keypoint_names[human_id].index('right_shoulder')],
            pnt_sc[coco_keypoint_names[human_id].index('left_shoulder')])

        mid_hip = (
            pnt[coco_keypoint_names[human_id].index('right_hip'), :2] +
            pnt[coco_keypoint_names[human_id].index('left_hip'), :2]) / 2
        mid_hip_sc = np.minimum(
            pnt_sc[coco_keypoint_names[human_id].index('right_hip')],
            pnt_sc[coco_keypoint_names[human_id].index('left_hip')])
        if (mid_shoulder_sc > thresh and
                pnt_sc[coco_keypoint_names[human_id].index('nose')] > thresh):
            y = [mid_shoulder[0],
                 pnt[coco_keypoint_names[human_id].index('nose'), 0]]
            x = [mid_shoulder[1],
                 pnt[coco_keypoint_names[human_id].index('nose'), 1]]
            line = ax.plot(x, y)
            plt.setp(
                line, color=colors[len(coco_point_skeleton)],
                linewidth=linewidth, alpha=0.7)
        if (mid_shoulder_sc > thresh and mid_hip_sc > thresh):
            y = [mid_shoulder[0], mid_hip[0]]
            x = [mid_shoulder[1], mid_hip[1]]
            line = ax.plot(x, y)
            plt.setp(
                line, color=colors[len(coco_point_skeleton) + 1],
                linewidth=linewidth, alpha=0.7)

    return ax
