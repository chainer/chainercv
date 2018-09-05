import numpy as np

exp_clip = np.log(1000 / 16)

from fpn.head import Head  # NOQA
from fpn.head import head_loss_post  # NOQA
from fpn.head import head_loss_pre  # NOQA
from fpn.faster_rcnn import FasterRCNN  # NOQA
from fpn.faster_rcnn_fpn_resnet import FasterRCNNFPNResNet101  # NOQA
from fpn.faster_rcnn_fpn_resnet import FasterRCNNFPNResNet50  # NOQA
from fpn.fpn import FPN  # NOQA
from fpn.manual_scheduler import ManualScheduler  # NOQA
from fpn.roi_align_2d import roi_align_2d  # NOQA
from fpn.rpn import RPN  # NOQA
from fpn.rpn import rpn_loss  # NOQA
from fpn.smooth_l1 import smooth_l1  # NOQA
