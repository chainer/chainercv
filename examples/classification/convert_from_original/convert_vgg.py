import chainer
from chainer.links import VGG16Layers as VGG16Layers_chainer

from chainercv.links import VGG16 as VGG16_cv


def main():
    chainer_model = VGG16Layers_chainer(pretrained_model='auto')
    cv_model = VGG16_cv(pretrained_model=None, n_class=1000)

    cv_model.conv1_1.conv.copyparams(chainer_model.conv1_1)

    # The pretrained weights are trained to accept BGR images.
    # Convert weights so that they accept RGB images.
    cv_model.conv1_1.conv.W.data[:] = cv_model.conv1_1.conv.W.data[:, ::-1]

    cv_model.conv1_2.conv.copyparams(chainer_model.conv1_2)
    cv_model.conv2_1.conv.copyparams(chainer_model.conv2_1)
    cv_model.conv2_2.conv.copyparams(chainer_model.conv2_2)
    cv_model.conv3_1.conv.copyparams(chainer_model.conv3_1)
    cv_model.conv3_2.conv.copyparams(chainer_model.conv3_2)
    cv_model.conv3_3.conv.copyparams(chainer_model.conv3_3)
    cv_model.conv4_1.conv.copyparams(chainer_model.conv4_1)
    cv_model.conv4_2.conv.copyparams(chainer_model.conv4_2)
    cv_model.conv4_3.conv.copyparams(chainer_model.conv4_3)
    cv_model.conv5_1.conv.copyparams(chainer_model.conv5_1)
    cv_model.conv5_2.conv.copyparams(chainer_model.conv5_2)
    cv_model.conv5_3.conv.copyparams(chainer_model.conv5_3)
    cv_model.fc6.copyparams(chainer_model.fc6)
    cv_model.fc7.copyparams(chainer_model.fc7)
    cv_model.fc8.copyparams(chainer_model.fc8)

    chainer.serializers.save_npz('vgg_from_caffe.npz', cv_model)


if __name__ == '__main__':
    main()
