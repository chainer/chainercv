import numpy as np

exp_clip = np.log(1000 / 16)

from chainercv.links.model.fpn.head import Head  # NOQA
from chainercv.links.model.fpn.head import head_loss_post  # NOQA
from chainercv.links.model.fpn.head import head_loss_pre  # NOQA
from chainercv.links.model.fpn.faster_rcnn import FasterRCNN  # NOQA
from chainercv.links.model.fpn.faster_rcnn_fpn_resnet import FasterRCNNFPNResNet101  # NOQA
from chainercv.links.model.fpn.faster_rcnn_fpn_resnet import FasterRCNNFPNResNet50  # NOQA
from chainercv.links.model.fpn.fpn import FPN  # NOQA
from chainercv.links.model.fpn.rpn import RPN  # NOQA
from chainercv.links.model.fpn.rpn import rpn_loss  # NOQA
from chainercv.links.model.fpn.smooth_l1 import smooth_l1  # NOQA
