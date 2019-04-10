# Examples of Faster R-CNN and related models using FPN [1, 2]

## Performance
MS COCO 2017 Val

| Backbone | Supervision | Original (Bbox) | Ours (Bbox) | Original (Mask) | Ours (Mask)| 
|:-:|:-:|:-:|:-:|:-:|:-:|
| FPN w/ ResNet50 | bbox | 36.7 % [3] | 37.1 % | | |
| FPN w/ ResNet101 | bbox | 39.4 % [3] | 39.5 % | | |
| FPN w/ ResNet50 | bbox + mask | 37.3 % [3] | 38.0 % | 33.7% [3] | 34.2 %|
| FPN w/ ResNet101 | bbox + mask | 39.4 % [3] | 40.4% | 35.6% [3] | 36.0% |

Scores are the mean of mean Average Precision (mmAP).

## Demo
If `faster_rcnn_*` is used as `--model`, the script conducts object detection.
If `mask_rcnn_*` is used as `--model`, the script conducts instance segmentation instead.
This demo downloads MS COCO pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--model faster_rcnn_fpn_resnet50|faster_rcnn_fpn_101|mask_rcnn_fpn_50|mask_rcnn_fpn_101] [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Evaluation
For object detection, use [`chainercv/examples/detection/eval_detection.py`](https://github.com/chainer/chainercv/blob/master/examples/detection) for evaluation.
For instance segmentation, use [`chainercv/examples/detection/eval_instance_segmentation.py`](https://github.com/chainer/chainercv/blob/master/examples/instance_segmentation) for evaluation.

## Train
You can train the model with the following code.
Note that this code requires `chainermn` module.
```
$ mpiexec -n <#gpu> python train_multi.py [--model faster_rcnn_fpn_resnet50|faster_rcnn_fpn_resnet101|mask_rcnn_fpn_resnet50|mask_rcnn_fpn_resnet101] [--batchsize <batchsize>]
```

## References
1. Tsung-Yi Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017
2. Kaiming He et al. "Mask R-CNN" ICCV 2017
3. [Detectron](https://github.com/facebookresearch/Detectron)
