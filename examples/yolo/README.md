# Examples of YOLO [1]

## Performance
PASCAL VOC2007 Test

| Model | Original | ChainerCV (weight conversion) |
|:-:|:-:|:-:|
| YOLOv3 | 80.2 % | 80.2 % |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

## Demo
Detect objects in an given image. This demo downloads Pascal VOC pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--gpu <gpu>] [--pretrained_model <model_path>] <image>.jpg
```

## Convert Darknet model
Convert `*.weights` to `*.npz`. YOLOv3 is supported.
Note that the number of classes should be specified if it is not 80 (the number of classes in COCO).
```
$ python darknet2npz [--n_fg_class <#fg_class>]  <source>.weights <target>.npz
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_voc07.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## References
1. Joseph Redmon et al. "YOLOv3: An Incremental Improvement" arXiv 2018.
