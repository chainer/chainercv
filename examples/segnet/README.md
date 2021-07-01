SegNet
======

# Preparation

Please create `class_weight.npy` using calc_weight.py first. Just run:

```
$ python calc_weight.py
```

# Start training

First, move to this directory (i.e., `examples/segnet`) and run:

```
$ python train.py [--gpu <gpu>]
```

PlotReport extension uses matplotlib. If you got `RuntimeError: Invalid DISPLAY variable` error on Linux environment, adding an environment variable specification is recommended:

```
$ MPLBACKEND=Agg python train.py [--gpu <gpu>]
```

## NOTE

- According to the original implementation, the authors performed LR flipping to the input images for data augmentation: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/dense_image_data_layer.cpp#L168-L175
- Chainer's LRN layer is different from Caffe's one in terms of the meaning of "alpha" argment, so we modified the Chainer's LRN default argment to make it same as Caffe's one: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/lrn_layer.cpp#L121

## Experimental settings

We used the completely same parameters for all settings.

| Implementation | Optimizer   | Learning rage | Momentum | Weight decay | Model code |
|:--------------:|:-----------:|:-------------:|:--------:|:------------:|:----------:|
| Ours      | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic.py](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/segnet/segnet_basic.py) |
| Original       | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic_train.prototxt](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt) |

# Quick Demo

Here is a quick demo using our pretrained weights. The pretrained model is automatically downloaded from the internet.

```
$ wget https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/test/0001TP_008550.png
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] 0001TP_008550.png
```

## Comparizon with the paper results

| Implementation | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:----------:|
| Ours      | 82.7 %          | **67.1 %**     | **49.4 %** |
| Original       | **82.8 %**      | 62.3%          | 46.3 %     |

The evaluation can be conducted using [`chainercv/examples/semantic_segmentation/eval_semantic_segmentation.py`](https://github.com/chainer/chainercv/blob/master/examples/semantic_segmentation).


# Reference

1. Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
2. Vijay Badrinarayanan, Ankur Handa and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling." arXiv preprint arXiv:1505.07293, 2015.
3. [Original implementation](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)
