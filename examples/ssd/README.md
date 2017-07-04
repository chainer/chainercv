# Examples of Single Shot Multibox Detector [1]

## Performance
PASCAL VOC2007 Test

| Model | Original | ChainerCV (weight conversion) | ChainerCV (train) |
|:-:|:-:|:-:|:-:|
| SSD300 | 77.5 % [2] | 77.8 % | 77.5 % |
| SSD512 | 79.5 % [2] | 79.7 % | - |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

## Demo
Detect objects in an given image. This demo downloads Pascal VOC pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--model ssd300|ssd512] [--gpu <gpu>] [--pretrained_model <model_path>] <image>.jpg
```

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`. Some layers are renamed to fit ChainerCV. SSD300 and SSD512 are supported.
```
$ python caffe2npz <source>.caffemodel <target>.npz
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_voc07.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## Train
You can train the model with the following code.
Note that this code requires `cv2` module.
```
$ python train.py [--gpu <gpu>]
```

## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
