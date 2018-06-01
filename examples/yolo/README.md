# Examples of YOLO [1, 2]

## Performance
PASCAL VOC2007 Test

| Model | Original | ChainerCV (weight conversion) |
|:-:|:-:|:-:|
| YOLOv2 | 75.8 % * | 75.8 % |
| YOLOv3 | 80.2 % | 80.2 % |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

\*: Although the original paper [1] reports 76.8 %, the darknet implementation and the provided weights achieved the lower score.
Similar issue is reported [here](https://github.com/AlexeyAB/darknet#how-to-calculate-map-on-pascalvoc-2007).

## Demo
Detect objects in an given image. This demo downloads Pascal VOC pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--model yolo_v2|yolo_v3] [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Convert Darknet model
Convert `*.weights` to `*.npz`. YOLOv2 and YOLOv3 are supported.
Note that the number of classes should be specified if it is not 80 (the number of classes in COCO).
```
$ python darknet2npz [--model yolo_v2|yolo_v3] [--n-fg-class <#fg_class>]  <source>.weights <target>.npz
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_voc07.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## References
1. Joseph Redmon et al. "YOLO9000: Better, Faster, Stronger" CVPR 2017.
2. Joseph Redmon et al. "YOLOv3: An Incremental Improvement" arXiv 2018.
