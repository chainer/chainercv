# Examples of Light-Head R-CNN for Object Detection [1]

## Performance
MS COCO 2014 minival

| Model | mAP@[0.5:0.95] (Original) |  mAP@[0.5:0.95] (ChainerCV) |
|:-:|:-:|:-:|
| Light-Head-RCNN ResNet101 | 39.6 % [1] / 40.0 % [2] | 39.3 % |

## Demo
Detect objects in an given image. This demo downloads MS COCO pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/detection/eval_coco.py`](https://github.com/chainer/chainercv/blob/master/examples/detection).

## Train
You can train the model with the following code.
Note that this code requires `chainermn` module.
```
$ mpiexec -n <#gpu> python train_multi.py [--batch-size <batch_size>]
```

## References
1. Zeming Li et al. "Light-Head R-CNN: In Defense of Two-Stage Object Detector" ArXiv 2017
2. [Light-Head RCNN](https://github.com/zengarden/light_head_rcnn)
