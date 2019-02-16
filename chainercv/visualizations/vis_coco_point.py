from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from chainercv.datasets import coco_point_names
from chainercv.visualizations.vis_image import vis_image


coco_point_skeleton = [
    [coco_point_names.index('left_eye'),
     coco_point_names.index('right_eye')],
    [coco_point_names.index('left_eye'),
     coco_point_names.index('nose')],
    [coco_point_names.index('right_eye'),
     coco_point_names.index('nose')],
    [coco_point_names.index('right_eye'),
     coco_point_names.index('right_ear')],
    [coco_point_names.index('left_eye'),
     coco_point_names.index('left_ear')],
    [coco_point_names.index('right_shoulder'),
     coco_point_names.index('right_elbow')],
    [coco_point_names.index('right_elbow'),
     coco_point_names.index('right_wrist')],
    [coco_point_names.index('left_shoulder'),
     coco_point_names.index('left_elbow')],
    [coco_point_names.index('left_elbow'),
     coco_point_names.index('left_wrist')],
    [coco_point_names.index('right_hip'),
     coco_point_names.index('right_knee')],
    [coco_point_names.index('right_knee'),
     coco_point_names.index('right_ankle')],
    [coco_point_names.index('left_hip'),
     coco_point_names.index('left_knee')],
    [coco_point_names.index('left_knee'),
     coco_point_names.index('left_ankle')],
    [coco_point_names.index('right_shoulder'),
     coco_point_names.index('left_shoulder')],
    [coco_point_names.index('right_hip'),
     coco_point_names.index('left_hip')]
]


def vis_coco_point(img, point, point_score, thresh=2, ax=None):
    from matplotlib import pyplot as plt

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(coco_point_skeleton) + 2)]

    # plt.autoscale(False)
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
                plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
            if s0 > thresh:
                ax.plot(
                    x0, y0, '.', color=colors[l],
                    markersize=3.0, alpha=0.7)
            if s1 > thresh:
                ax.plot(
                    x1, y1, '.', color=colors[l],
                    markersize=3.0, alpha=0.7)

        # for better visualization, add mid shoulder / mid hip
        mid_shoulder = (
            pnt[coco_point_names.index('right_shoulder'), :2] +
            pnt[coco_point_names.index('left_shoulder'), :2]) / 2
        mid_shoulder_sc = np.minimum(
            pnt_sc[coco_point_names.index('right_shoulder')],
            pnt_sc[coco_point_names.index('left_shoulder')])

        mid_hip = (
            pnt[coco_point_names.index('right_hip'), :2] +
            pnt[coco_point_names.index('left_hip'), :2]) / 2
        mid_hip_sc = np.minimum(
            pnt_sc[coco_point_names.index('right_hip')],
            pnt_sc[coco_point_names.index('left_hip')])
        if (mid_shoulder_sc > thresh and
                pnt_sc[coco_point_names.index('nose')] > thresh):
            y = [mid_shoulder[0], pnt[coco_point_names.index('nose'), 0]]
            x = [mid_shoulder[1], pnt[coco_point_names.index('nose'), 1]]
            line = ax.plot(x, y)
            plt.setp(
                line, color=colors[len(coco_point_skeleton)],
                linewidth=1.0, alpha=0.7)
        if (mid_shoulder_sc > thresh and mid_hip_sc > thresh):
            y = [mid_shoulder[0], mid_hip[0]]
            x = [mid_shoulder[1], mid_hip[1]]
            line = ax.plot(x, y)
            plt.setp(
                line, color=colors[len(coco_point_skeleton) + 1],
                linewidth=1.0, alpha=0.7)

    return ax


if __name__ == '__main__':
    data = np.load('vis_point.npz')
    img = data['img']
    point = data['point']
    point_score = data['point_score']
    # plt.imshow(img)
    vis_coco_point(img, point, point_score)
    plt.show()
