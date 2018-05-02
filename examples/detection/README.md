# Examples of detection models

## Supported models
- Faster R-CNN
- SSD300
- SSD512
- YOLOv3

For the details, please check the documents and examples of each model.

## Performance

### PASCAL VOC2007 Test

| Model | Train dataset | FPS | mAP (PASCAL VOC2007 metric) |
|:-:|:-:|:-:|:-:|
| Faster R-CNN | VOC2007 trainval | | 70.6 % |
| Faster R-CNN | VOC2007\&2012 trainval | | 74.7 % |
| SSD300 | VOC2007\&2012 trainval | | 77.8 % |
| SSD512 | VOC2007\&2012 trainval | | 79.7 % |
| YOLOv2 | VOC2007\&2012 trainval | | 75.8 % |
| YOLOv3 | VOC2007\&2012 trainval | | 80.2 % |

You can reproduce these scores by the following command.
```
$ python eval_voc07.py [--model faster_rcnn|ssd300|ssd512|yolo_v2|yolov3] [--pretrained_model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
```

## Visualization of models

![Visualization of models](https://cloud.githubusercontent.com/assets/2062128/26337670/44a2a202-3fb5-11e7-8b88-6eb9886a9915.png)
These images are included in PASCAL VOC2007 test.

You can generate these visualization results by the following command.
```
$ python visualuze_models.py
```

## Notes on writing your own evaluation code

Here is a list of important configurations to reproduce results.

+ `model.use_preset('evaluate')` configures postprocessing parameters for evaluation such as threshold for confidence score.
+ `DetectionVOCEvaluator` should be instantiated with `use_07_metric=True` (default is False), if evaluation is conducted on VOC 2007 test dataset.
+ When evaluating on VOC dataset, `VOCBboxDataset` should return information about difficulties of bounding boxes, as the evaluation metric expects that to be included.
The dataset returns it by setting `use_difficult=True` and `return_difficult=True`.
