# Examples of Feature Pyramid Networks for Object Detection [1]

## Performance
MS COCO 2017 Val

| Model | Original | ChainerCV |
|:-:|:-:|:-:|
| Faster-RCNN FPN ResNet50 | 36.7 % [2] | 37.1 % |
| Faster-RCNN FPN ResNet101 | 39.4 % [2] | 39.5 % |

Scores are the mean of mean Average Precision (mmAP).

## Demo
Detect objects in an given image. This demo downloads MS COCO pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--model faster_rcnn_fpn_resnet50|faster_rcnn_fpn_101] [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_detection.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## Train
You can train the model with the following code.
Note that this code requires `chainermn` module.
```
$ mpiexec -n <#gpu> python train_multi.py [--model faster_rcnn_fpn_resnet50|faster_rcnn_fpn_resnet101] [--batchsize <batchsize>]
```

## References
1. Tsung-Yi Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017
2. [Detectron](https://github.com/facebookresearch/Detectron)
