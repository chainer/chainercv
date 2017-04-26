from chainercv.links import FasterRCNNVGG

import numpy as np

import chainer.links.caffe.caffe_function as caffe


class _CaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path):
        super(_CaffeFunction, self).__init__(model_path)

    @caffe._layer('Python', None)
    @caffe._layer('ROIPooling', None)
    @caffe._layer('SmoothL1Loss', None)
    @caffe._layer('Silence', None)
    def _skip_layer(self, _):
        pass


def load_caffe(model_path, model):
    caffe_model = _CaffeFunction(model_path)

    model.feature.conv1_1.copyparams(caffe_model.conv1_1)
    model.feature.conv1_2.copyparams(caffe_model.conv1_2)

    model.feature.conv2_1.copyparams(caffe_model.conv2_1)
    model.feature.conv2_2.copyparams(caffe_model.conv2_2)

    model.feature.conv3_1.copyparams(caffe_model.conv3_1)
    model.feature.conv3_2.copyparams(caffe_model.conv3_2)
    model.feature.conv3_3.copyparams(caffe_model.conv3_3)

    model.feature.conv4_1.copyparams(caffe_model.conv4_1)
    model.feature.conv4_2.copyparams(caffe_model.conv4_2)
    model.feature.conv4_3.copyparams(caffe_model.conv4_3)

    model.feature.conv5_1.copyparams(caffe_model.conv5_1)
    model.feature.conv5_2.copyparams(caffe_model.conv5_2)
    model.feature.conv5_3.copyparams(caffe_model.conv5_3)

    model.head.fc6.copyparams(caffe_model.fc6)
    model.head.fc7.copyparams(caffe_model.fc7)

    model.head.cls_score.copyparams(caffe_model.cls_score)
    model.head.bbox_pred.copyparams(caffe_model.bbox_pred)

    model.rpn.rpn_conv_3x3.copyparams(getattr(caffe_model, 'rpn_conv/3x3'))
    model.rpn.rpn_cls_score.copyparams(caffe_model.rpn_cls_score)
    model.rpn.rpn_bbox_pred.copyparams(caffe_model.rpn_bbox_pred)


if __name__ == '__main__':
    import argparse
    from chainer import serializers
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffemodel', type=str,
                        default='VGG16_faster_rcnn_final.caffemodel')
    parser.add_argument('--output', type=str,
                        default='VGG16_faster_rcnn_final.npz')
    args = parser.parse_args()

    caffe_func = _CaffeFunction(args.caffemodel)
    model = FasterRCNNVGG()
    load_caffe(args.caffemodel, model)
    serializers.save_npz(args.output, model)
