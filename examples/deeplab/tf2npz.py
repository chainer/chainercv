import argparse

import chainer

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import DeepLabV3plusXception65

import tensorflow as tf

_n_class = {
    'voc': len(voc_semantic_segmentation_label_names),
    'cityscapes': len(cityscapes_semantic_segmentation_label_names),
    'ade20k': len(ade20k_semantic_segmentation_label_names),
}

_model_class = {
    'xception65': DeepLabV3plusXception65,
}


def load_param(param, weight, transpose=None):
    if isinstance(param, chainer.Variable):
        param = param.array

    if transpose is not None:
        weight = weight.transpose(transpose)

    param[:] = weight


def get_model(name, task):
    n_class = _n_class[task]
    model = _model_class[name](n_class, crop=(513, 513), scales=(1.0,),
                               flip=False, extractor_kwargs={},
                               aspp_kwargs={}, decoder_kwargs={})
    return model


def get_session(graph_path):
    # load graph
    with tf.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='deeplab',
            producer_op_list=None
        )

    graph_options = tf.GPUOptions(visible_device_list="0", allow_growth=True)
    config = config = tf.ConfigProto(gpu_options=graph_options)
    sess = tf.Session(graph=graph, config=config)
    return sess


def get_weightmap(model):
    weightmap = {}
    if model == 'xception65':
        weightmap[('feature_extractor', 'entryflow_conv1')] = (
            'Conv2DBNActiv', 'deeplab/xception_65/entry_flow/conv1_1')
        weightmap[('feature_extractor', 'entryflow_conv2')] = (
            'Conv2DBNActiv', 'deeplab/xception_65/entry_flow/conv1_2')
        weightmap[('feature_extractor', 'entryflow_block1')] = (
            'XceptionBlock', 'deeplab/xception_65/entry_flow/block1/unit_1')
        weightmap[('feature_extractor', 'entryflow_block2')] = (
            'XceptionBlock', 'deeplab/xception_65/entry_flow/block2/unit_1')
        weightmap[('feature_extractor', 'entryflow_block3')] = (
            'XceptionBlock', 'deeplab/xception_65/entry_flow/block3/unit_1')
        for i in range(1, 17):
            weightmap[('feature_extractor',
                       'middleflow_block{}'.format(i))] = (
                'XceptionBlock',
                'deeplab/xception_65/middle_flow/block1/unit_{}'.format(i))
        weightmap[('feature_extractor', 'exitflow_block1')] = (
            'XceptionBlock', 'deeplab/xception_65/exit_flow/block1/unit_1')
        weightmap[('feature_extractor', 'exitflow_block2')] = (
            'XceptionBlock', 'deeplab/xception_65/exit_flow/block2/unit_1')

        weightmap[('aspp', 'image_pooling_conv')] = (
            'Conv2DBNActiv', 'deeplab/image_pooling')
        weightmap[('aspp', 'conv1x1')] = ('Conv2DBNActiv', 'deeplab/aspp0')
        weightmap[('aspp', 'atrous1')] = (
            'SeparableConv2DBNActiv', 'deeplab/aspp1')
        weightmap[('aspp', 'atrous2')] = (
            'SeparableConv2DBNActiv', 'deeplab/aspp2')
        weightmap[('aspp', 'atrous3')] = (
            'SeparableConv2DBNActiv', 'deeplab/aspp3')
        weightmap[('aspp', 'proj')] = (
            'Conv2DBNActiv', 'deeplab/concat_projection')

        weightmap[('decoder', 'feature_proj')] = (
            'Conv2DBNActiv', 'deeplab/decoder/feature_projection0')
        weightmap[('decoder', 'conv1')] = (
            'SeparableConv2DBNActiv', 'deeplab/decoder/decoder_conv0')
        weightmap[('decoder', 'conv2')] = (
            'SeparableConv2DBNActiv', 'deeplab/decoder/decoder_conv1')
        weightmap[('decoder', 'conv_logits')] = (
            'Convolution2D', 'deeplab/logits/semantic')
    else:
        raise

    return weightmap


def resolve(weightmap):
    # resolve weightmap
    changed = True
    while changed:
        changed = False
        for key in list(weightmap.keys()):
            layer, op = weightmap.pop(key)
            if layer == 'Conv2DBNActiv':
                weightmap[key+('conv',)] = ('Convolution2D', op)
                weightmap[key+('bn',)] = (
                    'BatchNormalization', op + '/BatchNorm')
                changed = True
            elif layer == 'SeparableConv2DBNActiv':
                weightmap[key+('depthwise',)] = (
                    'Convolution2D_depthwise', op + '_depthwise')
                weightmap[key+('dw_bn',)] = (
                    'BatchNormalization', op + '_depthwise/BatchNorm')
                weightmap[key+('pointwise',)] = (
                    'Convolution2D', op + '_pointwise')
                weightmap[key+('pw_bn',)] = (
                    'BatchNormalization', op + '_pointwise/BatchNorm')
                changed = True
            elif layer == 'XceptionBlock':
                weightmap[key+('separable1',)] = (
                    'SeparableConv2DBNActiv',
                    op + '/xception_module/separable_conv1')
                weightmap[key+('separable2',)] = (
                    'SeparableConv2DBNActiv',
                    op + '/xception_module/separable_conv2')
                weightmap[key+('separable3',)] = (
                    'SeparableConv2DBNActiv',
                    op + '/xception_module/separable_conv3')
                weightmap[key+('conv',)] = (
                    'Conv2DBNActiv', op + '/xception_module/shortcut')
                changed = True
            else:
                weightmap[key] = (layer, op)
    return weightmap


def transfer(model, sess, weightmap):
    for key, (layer, op) in weightmap.items():
        link = model

        try:
            for sublink in key:
                link = link[sublink]
        except AttributeError:
            continue

        print('loading: {}'.format('/'.join(key)))

        input_dict = {}
        transpose = {}
        if layer == 'Convolution2D':
            input_dict['W'] = op + '/weights:0'
            input_dict['b'] = op + '/biases:0'
            transpose['W'] = (3, 2, 0, 1)
        elif layer == 'Convolution2D_depthwise':
            input_dict['W'] = op + '/depthwise_weights:0'
            transpose['W'] = (2, 3, 0, 1)
        elif layer == 'BatchNormalization':
            input_dict['gamma'] = op + '/gamma:0'
            input_dict['beta'] = op + '/beta:0'
            input_dict['avg_mean'] = op + '/moving_mean:0'
            input_dict['avg_var'] = op + '/moving_variance:0'
        else:
            raise ValueError('Invalid layer: {}'.format(layer))

        for k in list(input_dict.keys()):
            if not hasattr(link, k) or getattr(link, k) is None:
                input_dict.pop(k)

        weights = sess.run(input_dict)

        for k in input_dict:
            load_param(getattr(link, k), weights[k], transpose.get(k))


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('model', choices=list(_model_class.keys()))
    parser.add_argument('task', choices=list(_n_class.keys()))
    parser.add_argument('graph_path')
    parser.add_argument('output')
    args = parser.parse_args()

    # currently, xception65 is only implemented.
    # model_name = args.model
    model_name = 'xception65'
    model = get_model(model_name, args.task)
    sess = get_session(args.graph_path)
    weightmap = get_weightmap(model_name)
    weightmap = resolve(weightmap)

    transfer(model, sess, weightmap)
    chainer.serializers.save_npz(args.output, model)

    sess.close()


if __name__ == '__main__':
    main()
