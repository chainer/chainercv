# Example codes for Faster R-CNN

### Performance

| Training Setting | Evaluation | Reference Implementation | ChainerCV |
|:-:|:-:|:-:|:-:|
| VOC 2007 trainval | VOC 2007 test|  69.9 mAP [1] | 70.5 mAP |


### Demo

```
$ python demo.py [--gpu <gpu>] <image>.jpg
```

This example will automatically download a pretrained weights from the internet when executed.
A sample image to try the implementation can be found in the link below.

https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg


### Difference in the runtime behaviour from the original code

The bounding box follows integer bbox convention in the original implementation, whereas the ChainerCV implementation follows float bbox convention used in COCO.
The integer convention encodes right below vertex coordinates of bounding boxes by subtracting one from the ground truth, whereas the float convention does not.

On top of that, the anchors are not discretized in ChainerCV.


### References
This code is based on Caffe implementation by the original authors https://github.com/rbgirshick/py-faster-rcnn and Chainer a re-implementation https://github.com/mitmul/chainer-faster-rcnn .

1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.
