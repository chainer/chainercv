# Examples of detection models

## Supported models
- Faster R-CNN
- Faster R-CNN FPN ResNet50
- Faster R-CNN FPN ResNet101
- SSD300
- SSD512
- YOLOv2
- YOLOv2 tiny
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
| YOLOv2 tiny | VOC2007\&2012 trainval | | 53.5 % |
| YOLOv3 | VOC2007\&2012 trainval | | 80.2 % |

You can reproduce these scores by the following command.
```
$ python eval_detection.py --dataset voc [--model faster_rcnn|ssd300|ssd512|yolo_v2|yolo_v2_tiny|yolo_v3] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
```

### MS COCO2017 Val

| Model | Train dataset | FPS | mmAP |
|:-:|:-:|:-:|:-:|
| Faster R-CNN FPN ResNet50 | COCO2017 train | | 37.1 % |
| Faster R-CNN FPN ResNet101 | COCO2017 train | | 39.5 % |

You can reproduce these scores by the following command.
```
$ python eval_detection.py --dataset coco [--model faster_rcnn_fpn_resnet50|faster_rcnn_fpn_resnet101] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
```

## Visualization of models

![Visualization of models](https://user-images.githubusercontent.com/3014172/40634581-bb01f52a-6330-11e8-8502-ba3dacd81dc8.png)
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
