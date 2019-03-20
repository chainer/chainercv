import argparse

import numpy as np

import chainer

import tensorflow as tf

from chainercv.links import MobileNetV2


def load_expanded_conv(econv, expand_params, depthwise_params, project_params):
    if hasattr(econv, 'expand'):
        assert expand_params is not None
        c_to_p = [(econv.expand.conv, econv.expand.bn, expand_params)]
    else:
        assert expand_params is None
        c_to_p = []
    c_to_p.extend([(econv.depthwise.conv, econv.depthwise.bn,
                    depthwise_params), (econv.project.conv, econv.project.bn,
                                        project_params)])

    for conv, bn, params in c_to_p:
        init_conv_with_tf_weights(conv, params["weights"])
        init_bn_with_tf_params(bn, params["beta"], params["gamma"],
                               params["moving_mean"],
                               params["moving_variance"])


def init_conv_with_tf_weights(conv, weights, bias=None):
    # Shifting input and output channel dimensions.
    weights = weights.transpose((3, 2, 0, 1))
    if conv.W.shape != weights.shape:  # for depthwise conv
        weights = weights.transpose((1, 0, 2, 3))
    conv.W.data[:] = weights.data[:]

    if bias is not None:
        conv.b.data[:] = bias.data[:]


def init_bn_with_tf_params(bn, beta, gamma, moving_mean, moving_variance):
    beta = beta.flatten().astype(chainer.config.dtype)
    bn.beta.initializer = chainer.initializers.Constant(
        beta, dtype=chainer.config.dtype)
    bn.beta.initialize(shape=beta.shape)
    gamma = gamma.flatten().astype(chainer.config.dtype)
    bn.gamma.initializer = chainer.initializers.Constant(
        gamma, dtype=chainer.config.dtype)
    bn.gamma.initialize(shape=gamma.shape)
    bn.avg_mean = moving_mean.flatten().astype(chainer.config.dtype)
    bn.avg_var = moving_variance.flatten().astype(chainer.config.dtype)


def load_mobilenetv2_from_tensorflow_checkpoint(model, checkpoint_filename):
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_filename)

    # Loading weights for the expanded convolutions.
    tf_scope_to_expanded_conv = {
        "MobilenetV2/expanded_conv": model.expanded_conv,
    }
    for i in range(16):
        tf_scope_to_expanded_conv["MobilenetV2/expanded_conv_{}".format(
            i + 1)] = getattr(model, "expanded_conv_{}".format(i + 1))
    for tf_scope, expanded_conv in tf_scope_to_expanded_conv.items():
        print("Loading weights for %s" % tf_scope)
        # Expand convolution parameters
        if hasattr(expanded_conv, 'expand'):
            expand_params = {
                "weights":
                ckpt_reader.get_tensor(tf_scope + '/expand/weights'),
                "beta":
                ckpt_reader.get_tensor(tf_scope + '/expand/BatchNorm/beta'),
                "gamma":
                ckpt_reader.get_tensor(tf_scope + '/expand/BatchNorm/gamma'),
                "moving_mean":
                ckpt_reader.get_tensor(
                    tf_scope + '/expand/BatchNorm/moving_mean'),
                "moving_variance":
                ckpt_reader.get_tensor(
                    tf_scope + '/expand/BatchNorm/moving_variance')
            }
        else:
            print("Skipping expanded convolution for {}".format(tf_scope))
            expand_params = None
        # Depthwise convolution parameters
        depthwise_params = {
            "weights":
            ckpt_reader.get_tensor(tf_scope + '/depthwise/depthwise_weights'),
            "beta":
            ckpt_reader.get_tensor(tf_scope + '/depthwise/BatchNorm/beta'),
            "gamma":
            ckpt_reader.get_tensor(tf_scope + '/depthwise/BatchNorm/gamma'),
            "moving_mean":
            ckpt_reader.get_tensor(
                tf_scope + '/depthwise/BatchNorm/moving_mean'),
            "moving_variance":
            ckpt_reader.get_tensor(
                tf_scope + '/depthwise/BatchNorm/moving_variance')
        }

        # Project convolution parameters
        project_params = {
            "weights":
            ckpt_reader.get_tensor(tf_scope + '/project/weights'),
            "beta":
            ckpt_reader.get_tensor(tf_scope + '/project/BatchNorm/beta'),
            "gamma":
            ckpt_reader.get_tensor(tf_scope + '/project/BatchNorm/gamma'),
            "moving_mean":
            ckpt_reader.get_tensor(
                tf_scope + '/project/BatchNorm/moving_mean'),
            "moving_variance":
            ckpt_reader.get_tensor(
                tf_scope + '/project/BatchNorm/moving_variance'),
        }
        load_expanded_conv(
            expanded_conv,
            expand_params=expand_params,
            depthwise_params=depthwise_params,
            project_params=project_params,
        )
    # Similarly loading the vanilla convolutions.
    # Initial convolution
    init_conv_with_tf_weights(
        model.conv.conv,
        weights=ckpt_reader.get_tensor('MobilenetV2/Conv/weights'))
    init_bn_with_tf_params(
        model.conv.bn,
        beta=ckpt_reader.get_tensor('MobilenetV2/Conv/BatchNorm/beta'),
        gamma=ckpt_reader.get_tensor('MobilenetV2/Conv/BatchNorm/gamma'),
        moving_mean=ckpt_reader.get_tensor(
            'MobilenetV2/Conv/BatchNorm/moving_mean'),
        moving_variance=ckpt_reader.get_tensor(
            'MobilenetV2/Conv/BatchNorm/moving_variance'))
    # Final convolution before dropout (conv1)
    init_conv_with_tf_weights(
        model.conv1.conv,
        weights=ckpt_reader.get_tensor('MobilenetV2/Conv_1/weights'))
    init_bn_with_tf_params(
        model.conv1.bn,
        beta=ckpt_reader.get_tensor('MobilenetV2/Conv_1/BatchNorm/beta'),
        gamma=ckpt_reader.get_tensor('MobilenetV2/Conv_1/BatchNorm/gamma'),
        moving_mean=ckpt_reader.get_tensor(
            'MobilenetV2/Conv_1/BatchNorm/moving_mean'),
        moving_variance=ckpt_reader.get_tensor(
            'MobilenetV2/Conv_1/BatchNorm/moving_variance'))
    # Logits convolution
    init_conv_with_tf_weights(
        model.logits_conv,
        weights=ckpt_reader.get_tensor(
            'MobilenetV2/Logits/Conv2d_1c_1x1/weights'),
        bias=ckpt_reader.get_tensor('MobilenetV2/Logits/Conv2d_1c_1x1/biases'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name', choices=('mobilenetv2', ), default='mobilenetv2')
    parser.add_argument('pretrained_model')
    parser.add_argument('--n_class', type=int, default=1001)
    parser.add_argument('--depth_multiplier', type=float, default=1.0)
    parser.add_argument('output', nargs='?', default=None)
    args = parser.parse_args()

    model = MobileNetV2(args.n_class, depth_multiplier=args.depth_multiplier)
    load_mobilenetv2_from_tensorflow_checkpoint(model, args.pretrained_model)

    if args.output is None:
        output = '{}_{}_imagenet_convert.npz'.format(args.model_name, args.depth_multiplier)
    else:
        output = args.output
    chainer.serializers.save_npz(output, model)
    print("output: ", output)


if __name__ == '__main__':
    main()
