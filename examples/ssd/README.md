# Examples of Single Shot Multibox Detector [1]

## Performance
PASCAL VOC2007 Test

| Model | Original | Ours (weight conversion) | Ours (train) |
|:-:|:-:|:-:|:-:|
| SSD300 | 77.5 % [2] | 77.8 % | 77.5 % / 77.6 % (4 GPUs) |
| SSD512 | 79.5 % [2] | 79.7 % | 80.1 % * / 80.5 % (4 GPUs) |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

\*: We set batchsize to 24 because of memory limitation. The original paper used 32.

## Demo
Detect objects in an given image. This demo downloads Pascal VOC pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--model ssd300|ssd512] [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`. Some layers are renamed to fit ChainerCV. SSD300 and SSD512 are supported.
```
$ python caffe2npz.py <source>.caffemodel <target>.npz
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_detection.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## Train
You can train the model with the following code.
Note that this code requires `cv2` module.
```
$ python train.py [--model ssd300|ssd512] [--batchsize <batchsize>] [--gpu <gpu>]
```

If you want to use multiple GPUs, use `train_multi.py`.
Note that this code requires `chainermn` module.
```
$ mpiexec -n <#gpu> python train_multi.py [--model ssd300|ssd512] [--batchsize <batchsize>] [--test-batchsize <batchsize>]
```

You can download weights that were trained by ChainerCV.
- [SSD300](https://chainercv-models.preferred.jp/ssd300_voc0712_trained_2017_08_08.npz)
- [SSD512](https://chainercv-models.preferred.jp/ssd512_voc0712_trained_2017_08_08.npz)

## References
1. Wei Liu et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
