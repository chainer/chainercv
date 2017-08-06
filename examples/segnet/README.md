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
| ChainerCV      | 81.08 % (2.44%)   | **65.92 %** (1.11%)  | **47.81 %** (2.52%) |
| Official       | **82.8 %**      | 62.3%          | 46.3 %     |

The above values of the official implementation is found here: [Getting Started with SegNet](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

For the scores reported for ChainerCV, they are the means of five trials.
The standard deviations are in parentheses.



## Trained weights

Here are links to the weights trained using ChainerCV.

##### SegNet traiend with CamVid

| Trial | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:---------------:|
| [1](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/segnet_camvid_trained_2017_08_06_trial_0.npz)  | 78.33 % | 65.45 % | 45.81% |
|[2](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/segnet_camvid_trained_2017_08_06_trial_1.npz) | 77.88 % | 64.22% | 43.86%|
|[3](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/segnet_camvid_trained_2017_08_06_trial_2.npz) | 83.18 % | 67.24% | 50.08%|
|[4](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/segnet_camvid_trained_2017_08_06_trial_3.npz) | 83.34 % | 65.64% | 49.93%|
|[5](https://github.com/yuyu2172/share-weights/releases/download/0.0.2/segnet_camvid_2017_05_28.npz) | 82.66% | 67.05 % | 49.39%|


# Reference

1. Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
2. Vijay Badrinarayanan, Ankur Handa and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling." arXiv preprint arXiv:1505.07293, 2015.
