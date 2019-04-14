# Examples of instance segmentation models

## Supported models

- FCIS ResNet101

For the details, please check the documents and examples of each model.

## Performance

### SBD Test

| Model | FPS | mAP@0.5 | mAP@0.7 |
|:-:|:-:|:-:|:-:|
| FCIS ResNet101 | | 64.1 % | 51.2 % |

You can reproduce these scores by the following command.

```bash
$ python eval_sbd.py [--model fcis_resnet101] [--pretrained-model <model_path>] [--gpu <gpu>]
```

### COCO Test

| Model | FPS | mAP/iou@[0.5:0.95] | mAP/iou@0.5 | mAP/iou@[0.5:0.95] \(small) | mAP/iou@[0.5:0.95] \(medium) | mAP/iou@[0.5:0.95] \(large) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FCIS ResNet101 | | 24.3 % | 42.6 % | 6.0 % | 24.9 % | 42.8% |

You can reproduce these scores by the following command.

```bash
$ python eval_coco.py [--model fcis_resnet101] [--pretrained-model <model_path>] [--gpu <gpu>]
```

## Notes on writing your own evaluation code

Here is a list of important configurations to reproduce results.

+ `model.use_preset('evaluate')` configures postprocessing parameters for evaluation such as threshold for confidence score.
+ `InstanceSegmentationSBDEvaluator` should be instantiated with `use_07_metric=True` (default is False).

