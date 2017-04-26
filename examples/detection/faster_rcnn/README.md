# WIP

```
python setup.py build_ext --inplace
```

Most of the work is based on https://github.com/mitmul/chainer-faster-rcnn .

In this project following functionalities are added on top of the above mentioned work.

+ Visualization extension (`chainer_cv.extensions.DetectionVisReport`)

+ PASCAL VOC Detection 2007/2012 datasets with automatic download functionality.

+ Use a pretrained model of VGG with automatic download functionality.




# Test code


```
wget https://dl.dropboxusercontent.com/u/2498135/faster-rcnn/VGG16_faster_rcnn_final.model

python test_faster_rcnn.py
```
