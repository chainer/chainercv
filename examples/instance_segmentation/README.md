# Examples of instance segmentation models

## Supported models

- FCIS ResNet101
- Mask R-CNN FPN w/ ResNet50
- Mask R-CNN FPN w/ ResNet101

For the details, please check the documents and examples of each model.

## Performance

### SBD Test

| Model | FPS | mAP@0.5 | mAP@0.7 |
|:-:|:-:|:-:|:-:|
| FCIS ResNet101 | | 64.1 % | 51.2 % |

You can reproduce these scores by the following command.

```
$ python eval_instance_segmentation.py --dataset sbd [--model fcis_resnet101] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_instance_segmentation.py --dataset sbd [--model fcis_resnet101] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```

### COCO Test

| Model | FPS | mAP/iou@[0.5:0.95] | mAP/iou@[0.5:0.95] \(small) | mAP/iou@[0.5:0.95] \(medium) | mAP/iou@[0.5:0.95] \(large) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| FCIS ResNet101 | | 24.3 % | 6.0 % | 24.9 % | 42.8% |
| Mask R-CNN FPN w/ ResNet50 | | 34.2 % | 15.6 % | 36.9 % | 50.8% |
| Mask R-CNN FPN w/ ResNet101 | | 36.0 % | 16.5 % | 39.2 % | 53.8% |

You can reproduce these scores by the following command.

```
$ python eval_instance_segmentation.py --dataset coco [--model fcis_resnet101|mask_rcnn_fpn_resnet50|mask_rcnn_fpn_resnet101] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_instance_segmentation.py --dataset coco [--model fcis_resnet101|mask_rcnn_fpn_resnet50|mask_rcnn_fpn_resnet101] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```

## Notes on writing your own evaluation code

Here is a list of important configurations to reproduce results.

+ `model.use_preset('evaluate')` configures postprocessing parameters for evaluation such as threshold for confidence score.
+ `InstanceSegmentationSBDEvaluator` should be instantiated with `use_07_metric=True` (default is False).

