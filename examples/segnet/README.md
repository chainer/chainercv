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
python train.py [--gpu <gpu>]
```

## NOTE

- According to the original implementation, the authors performed LR flipping to the input images for data augmentation: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/dense_image_data_layer.cpp#L168-L175
- Chainer's LRN layer is different from Caffe's one in terms of the meaning of "alpha" argment, so we modified the Chainer's LRN default argment to make it same as Caffe's one: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/lrn_layer.cpp#L121

## Experimental settings

We used the completely same parameters for all settings.

| Implementation | Optimizer   | Learning rage | Momentum | Weight decay | Model code |
|:--------------:|:-----------:|:-------------:|:--------:|:------------:|:----------:|
| ChainerCV      | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic.py](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/segnet/segnet_basic.py) |
| Official       | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic_train.prototxt](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt) |

# Quick Demo

Here is a quick demo using our pretrained weights. The pretrained model is automatically downloaded from the internet.

```
wget https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/test/0001TP_008550.png
python demo.py [--gpu <gpu>] [--pretrained_model <model_path>] 0001TP_008550.png
```


# Evaluation

The trained weights to replicate the same results as below is here: [model_iteration-16000](https://www.dropbox.com/s/exas66necaqbxyw/model_iteration-16000).

```
python eval_camvid.py [--gpu <gpu>] [--pretrained_model <model_path>] [--batchsize <batchsize>]
```


# Results

Once you execute the above evaluation script, you will see the values of intersection over union (IoU) for each class, the mean IoU over all the classes, class average accuracy, and global average accuracy like this:

```
                    Sky : 0.8790
               Building : 0.6684
                   Pole : 0.1923
                   Road : 0.8739
               Pavement : 0.6421
                   Tree : 0.6227
             SignSymbol : 0.1893
                  Fence : 0.2137
                    Car : 0.6355
             Pedestrian : 0.2739
              Bicyclist : 0.2415
==================================
               mean IoU : 0.4939
 Class average accuracy : 0.6705
Global average accuracy : 0.8266
```

## Comparizon with the paper results

| Implementation | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:----------:|
| ChainerCV      | 82.7 %          | **67.1 %**     | **49.4 %** |
| Official       | **82.8 %**      | 62.3%          | 46.3 %     |

The above values of the official implementation is found here: [Getting Started with SegNet](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

# Reference

1. Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017. 
2. Vijay Badrinarayanan, Ankur Handa and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling." arXiv preprint arXiv:1505.07293, 2015.
