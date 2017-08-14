Pyramid Scene Parsing Network
=============================

This is an unofficial implementation of Pyramid Scene Parsing Network (PSPNet) in Chainer.

# Inference using converted weights

## Requirement

- Python 3.4.4+
    - Chainer 3.0.0b1+
    - ChainerCV 0.6.0+
    - Matplotlib 2.0.0+
    - CuPy 2.0.0b1+
    - tqdm 4.11.0+

## Run demo.py

```
$ python demo.py -g 0 -m cityscapes -f aachen_000000_000019_leftImg8bit.png
```

---

# Convert weights by yourself

**Caffe is NOT needed** to convert `.caffemodel` to Chainer model. Use `caffe_pb2.py`.

## Requirement

- Python 3.4.4+
    - protobuf 3.2.0+
    - Chainer 3.0.0b1+
    - NumPy 1.12.0+

## 1. Download the original weights

Please download the weights below from the author's repository:

- pspnet50\_ADE20K.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCN1R3QnUwQ0hoMTA)
- pspnet101\_VOC2012.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCNVhETE5vVUdMYk0)
- pspnet101\_cityscapes.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU)

**and then put them into `weights` directory.**

## 2. Convert weights

```
$ python convert.py
```
