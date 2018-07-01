import argparse
import chainer
import mxnet as mx

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.experimental.links import FCISResNet101


def main():
    parser = argparse.ArgumentParser(
        description='Script to convert mxnet params to chainer npz')
    parser.add_argument('--param-file')
    parser.add_argument('--process', action='store_true')
    parser.add_argument(
        '--out', '-o', type=str, default='fcis_resnet101_sbd_converted.npz')
    args = parser.parse_args()

    model = FCISResNet101(
        n_fg_class=len(sbd_instance_segmentation_label_names),
        pretrained_model=None)
    params = mx.nd.load(args.param_file)
    print('mxnet param is loaded: {}'.format(args.param_file))
    print('start conversion')
    if args.process:
        tests = [k for k in params.keys() if k.endswith('_test')]
        for test in tests:
            params[test.replace('_test', '')] = params.pop(test)
    model = convert(model, params)
    print('finish conversion')
    print('saving to {}'.format(args.out))
    chainer.serializers.save_npz(args.out, model)


def convert(model, params):
    finished_keys = []
    for key, value in params.items():
        value = value.asnumpy()
        param_type, param_name = key.split(':')
        if param_type == 'arg':
            if param_name.endswith('_test'):
                continue
            elif param_name.startswith('rpn'):
                if param_name == 'rpn_bbox_pred_bias':
                    value = value.reshape((9, 4))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((36, ))
                    assert model.rpn.loc.b.shape == value.shape
                    model.rpn.loc.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'rpn_bbox_pred_weight':
                    value = value.reshape((9, 4, 512, 1, 1))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((36, 512, 1, 1))
                    assert model.rpn.loc.W.shape == value.shape
                    model.rpn.loc.W.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'rpn_cls_score_bias':
                    value = value.reshape((2, 9))
                    value = value.transpose((1, 0))
                    value = value.reshape((18, ))
                    assert model.rpn.score.b.shape == value.shape
                    model.rpn.score.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'rpn_cls_score_weight':
                    value = value.reshape((2, 9, 512, 1, 1))
                    value = value.transpose((1, 0, 2, 3, 4))
                    value = value.reshape((18, 512, 1, 1))
                    assert model.rpn.score.W.shape == value.shape
                    model.rpn.score.W.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'rpn_conv_3x3_bias':
                    assert model.rpn.conv1.b.shape == value.shape
                    model.rpn.conv1.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'rpn_conv_3x3_weight':
                    assert model.rpn.conv1.W.shape == value.shape
                    model.rpn.conv1.W.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('conv1'):
                if param_name == 'conv1_weight':
                    assert model.extractor.conv1.conv.W.shape \
                        == value.shape
                    model.extractor.conv1.conv.W.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('bn_conv1'):
                if param_name == 'bn_conv1_beta':
                    assert model.extractor.conv1.bn.beta.shape \
                        == value.shape
                    model.extractor.conv1.bn.beta.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'bn_conv1_gamma':
                    assert model.extractor.conv1.bn.gamma.shape \
                        == value.shape
                    model.extractor.conv1.bn.gamma.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('fcis'):
                if param_name == 'fcis_bbox_bias':
                    value = value.reshape((2, 4, 7 * 7))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((392, ))
                    assert model.head.ag_loc.b.shape == value.shape
                    model.head.ag_loc.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'fcis_bbox_weight':
                    value = value.reshape((2, 4, 7 * 7, 1024, 1, 1))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((392, 1024, 1, 1))
                    assert model.head.ag_loc.W.shape == value.shape
                    model.head.ag_loc.W.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'fcis_cls_seg_bias':
                    assert model.head.cls_seg.b.shape == value.shape
                    model.head.cls_seg.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'fcis_cls_seg_weight':
                    assert model.head.cls_seg.W.shape == value.shape
                    model.head.cls_seg.W.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('conv_new_1'):
                if param_name == 'conv_new_1_bias':
                    assert model.head.conv1.b.shape == value.shape
                    model.head.conv1.b.array[:] = value
                    finished_keys.append(key)
                elif param_name == 'conv_new_1_weight':
                    assert model.head.conv1.W.shape == value.shape
                    model.head.conv1.W.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('res'):
                block_name, branch_name, prm_name = param_name.split('_')
                resblock_name = block_name[:4]
                resblock = getattr(model.extractor, resblock_name)

                if block_name[4:] == 'a':
                    blck_name = block_name[4:]
                elif block_name[4:] == 'b':
                    blck_name = 'b1'
                elif block_name[4:].startswith('b'):
                    blck_name = block_name[4:]
                elif block_name[4:] == 'c':
                    blck_name = 'b2'
                block = getattr(resblock, blck_name)

                if branch_name == 'branch1':
                    conv_bn_name = 'residual_conv'
                elif branch_name == 'branch2a':
                    conv_bn_name = 'conv1'
                elif branch_name == 'branch2b':
                    conv_bn_name = 'conv2'
                elif branch_name == 'branch2c':
                    conv_bn_name = 'conv3'
                conv_bn = getattr(block, conv_bn_name)

                if prm_name == 'weight':
                    assert conv_bn.conv.W.shape == value.shape
                    conv_bn.conv.W.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('bn'):
                block_name, branch_name, prm_name = param_name.split('_')
                resblock_name = 'res{}'.format(block_name[2])
                resblock = getattr(model.extractor, resblock_name)

                if block_name[3:] == 'a':
                    blck_name = block_name[3:]
                elif block_name[3:] == 'b':
                    blck_name = 'b1'
                elif block_name[3:].startswith('b'):
                    blck_name = block_name[3:]
                elif block_name[3:] == 'c':
                    blck_name = 'b2'
                block = getattr(resblock, blck_name)

                if branch_name == 'branch1':
                    conv_bn_name = 'residual_conv'
                elif branch_name == 'branch2a':
                    conv_bn_name = 'conv1'
                elif branch_name == 'branch2b':
                    conv_bn_name = 'conv2'
                elif branch_name == 'branch2c':
                    conv_bn_name = 'conv3'
                conv_bn = getattr(block, conv_bn_name)

                if prm_name == 'beta':
                    assert conv_bn.bn.beta.shape == value.shape
                    conv_bn.bn.beta.array[:] = value
                    finished_keys.append(key)
                elif prm_name == 'gamma':
                    assert conv_bn.bn.gamma.shape == value.shape
                    conv_bn.bn.gamma.array[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            else:
                print('param: {} is not converted'.format(key))
        elif param_type == 'aux':
            if param_name.endswith('_test'):
                continue
            elif param_name.startswith('bn_conv1'):
                if param_name == 'bn_conv1_moving_mean':
                    assert model.extractor.conv1.bn.avg_mean.shape \
                        == value.shape
                    model.extractor.conv1.bn.avg_mean[:] = value
                    finished_keys.append(key)
                elif param_name == 'bn_conv1_moving_var':
                    assert model.extractor.conv1.bn.avg_var.shape \
                        == value.shape
                    model.extractor.conv1.bn.avg_var[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            elif param_name.startswith('bn'):
                block_name, branch_name, _, prm_name = \
                    param_name.split('_')
                resblock_name = 'res{}'.format(block_name[2])
                resblock = getattr(model.extractor, resblock_name)

                if block_name[3:] == 'a':
                    blck_name = block_name[3:]
                elif block_name[3:] == 'b':
                    blck_name = 'b1'
                elif block_name[3:].startswith('b'):
                    blck_name = block_name[3:]
                elif block_name[3:] == 'c':
                    blck_name = 'b2'
                block = getattr(resblock, blck_name)

                if branch_name == 'branch1':
                    conv_bn_name = 'residual_conv'
                elif branch_name == 'branch2a':
                    conv_bn_name = 'conv1'
                elif branch_name == 'branch2b':
                    conv_bn_name = 'conv2'
                elif branch_name == 'branch2c':
                    conv_bn_name = 'conv3'
                conv_bn = getattr(block, conv_bn_name)

                if prm_name == 'mean':
                    assert conv_bn.bn.avg_mean.shape == value.shape
                    conv_bn.bn.avg_mean[:] = value
                    finished_keys.append(key)
                elif prm_name == 'var':
                    assert conv_bn.bn.avg_var.shape == value.shape
                    conv_bn.bn.avg_var[:] = value
                    finished_keys.append(key)
                else:
                    print('param: {} is not converted'.format(key))
            else:
                print('param: {} is not converted'.format(key))
        else:
            print('param: {} is not converted'.format(key))

    return model


if __name__ == '__main__':
    main()
