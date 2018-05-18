# Examples of Single Shot Multibox Detector [1]

## Performance
PASCAL VOC2007 Test

| Model | Original | ChainerCV (weight conversion) | ChainerCV (train) |
|:-:|:-:|:-:|:-:|
| SSD300 | 77.5 % [2] | 77.8 % | 77.5 % / 77.3 % (4 GPUs) |
| SSD512 | 79.5 % [2] | 79.7 % | 80.1 % * / 79.9 % (4 GPUs) |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

\*: We set batchsize to 24 because of memory limitation. The original paper used 32.

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
$ python train.py [--model ssd300|ssd512] [--batchsize <batchsize>] [--gpu <gpu>]
```
Note that this training process sometimes stucks due to `cv2` issue.
For the details and workaround, please see [Chainer's Tips and FAQs](https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator).

If you want to use multiple GPUs, use `train_multi.py`.
Note that this code requires `chainermn` module.
```
$ mpi4exec -n <#gpu> python train_multi.py [--model ssd300|ssd512] [--batchsize <batchsize>] [--test_batchsize <batchsize>]
```

You can download weights that were trained by ChainerCV.
- [SSD300](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/ssd300_voc0712_trained_2017_08_08.npz)
- [SSD512](https://github.com/yuyu2172/share-weights/releases/download/0.0.4/ssd512_voc0712_trained_batchsize_24_2017_08_08.npz)

## References
1. Wei Liu et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
