# Examples of Single Shot Multibox Detector

## Demo
Detect objects in an given image. This demo downloads Pascal VOC pretrained model automatically.
```
$ python demo.py [--model ssd300|ssd512] <image>.jpg
```

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`. Some layers are renamed to fit ChainerCV. SSD300 and SSD512 are supported.
```
$ python caffe2npz <source>.caffemodel <target>.npz
```
