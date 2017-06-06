# Examples of detection models

## Supported models
- Faster R-CNN
- SSD300
- SSD512

For the details, please check the documents and examples of each model.

## Performance

### PASCAL VOC2007 Test

| Model | Train dataset | FPS | mAP (PASCAL VOC2007 metric) |
|:-:|:-:|:-:|:-:|
| Faster R-CNN | VOC2007 trainval | | 70.5 % |
| SSD300 | VOC2007\&2012 trainval | | 77.8 % |
| SSD512 | VOC2007\&2012 trainval | | 79.7 % |

You can reproduce these scores by the following command.
```
$ python eval_voc07.py [--model faster_rcnn|ssd300|ssd512] [--pretrained_model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
```

## Visualization of models

![Visualization of models](https://cloud.githubusercontent.com/assets/2062128/26337670/44a2a202-3fb5-11e7-8b88-6eb9886a9915.png)
These images are included in PASCAL VOC2007 test.

You can generate these visualization results by the following command.
```
$ python visualuze_models.py
```
