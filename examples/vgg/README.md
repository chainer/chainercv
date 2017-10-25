# VGG

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`.

```
$ python caffe2npz.py <source>.caffemodel <target>.npz
```

The pretrained `.caffemodel` for VGG-16 can be downloaded from here.
http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
