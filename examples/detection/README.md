# Examples of detection links

## Supported models
- Faster R-CNN
- SSD300
- SSD512

For the details, please check the documents and examples of each model.

## Performance

### PASCAL VOC2007 Test

| Model | FPS | mAP (PASCAL VOC2007 metric) |
|:-:|:-:|:-:|
| Faster R-CNN | | 70.5 % |
| SSD300 | | 77.8 % |
| SSD512 | | 79.7 % |

You can reproduce these scores by the following command.
```
$ python eval_voc07.py [--model faster_rcnn|ssd300|ssd512] [--gpu <gpu>]
```
