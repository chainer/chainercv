SegNet
======

# Preparation

Please create `class_weight.npy` using calc_weight.py first. Just run:

```
python calc_weight.py
```

# Start training

First, move to this directory (i.e., `examples/segnet`) and run:

```
CHAINER_SEED=2017 CHAINER_TYPE_CHECK=0 python -OO -W ignore train.py --gpu 0
```

## NOTE

- According to the original implementation, the authors performed LR flipping to the input images for data augmentation: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/dense_image_data_layer.cpp#L168-L175
- Chainer's LRN layer is different from Caffe's one in terms of the meaning of "alpha" argment, so we modified the Chainer's LRN default argment to make it same as Caffe's one: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/lrn_layer.cpp#L121

## Experimental settings

We used the completely same parameters for all settings.

| Implementation | Optimizer   | Learning rage | Momentum | Weight decay | Model code |
|:--------------:|:-----------:|:-------------:|:--------:|:------------:|:----------:|
| ChainerCV      | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic.py](https://github.com/pfnet/chainercv/tree/master/chainercv/links/model/segnet/segnet_basic.py) |
| Official       | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic_train.prototxt](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt) |

# Quick Demo

Here is a quick demo using our pretrained weights.

```
wget https://www.dropbox.com/s/exas66necaqbxyw/model_iteration-16000
wget https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/test/0001TP_008550.png
python demo.py 0001TP_008550.png model_iteration-16000
```

# Evaluation

The trained weights to replicate the same results as below is here: [model_iteration-16000](https://www.dropbox.com/s/exas66necaqbxyw/model_iteration-16000).

```
bash evaluate.sh [GPU ID] [MODEL SNAPSHOT]
```

e.g.,

```
bash evaluate.sh 0 result/2017-05-27_02-18-07/model_iteration-16000
```

# Results

Once you execute the above evaluation script, you will see the values of intersection over union (IoU) for each class, the mean IoU over all the classes, class average accuracy, and global average accuracy like this:

```
                    Sky : 0.8675
               Building : 0.6487
                   Pole : 0.1795
           Road_marking : 0.8493
                   Road : 0.6230
               Pavement : 0.6051
                   Tree : 0.1813
             SignSymbol : 0.1901
              Fence,Car : 0.5924
             Pedestrian : 0.2402
              Bicyclist : 0.2111
==================================
               mean IoU : 0.4716
 Class average accuracy : 0.6705
Global average accuracy : 0.8266
```

## Comparizon with the paper results

| Implementation | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:----------:|
| ChainerCV      | 82.7 %          | **67.1 %**     | **47.2 %** |
| Official       | **82.8 %**      | 62.3%          | 46.3 %     |

The above values of the official implementation is found here: [Getting Started with SegNet](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

# Reference

- Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017. 
- Vijay Badrinarayanan, Ankur Handa and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling." arXiv preprint arXiv:1505.07293, 2015.
