# Squeeze-and-Excitation Networks (SENet)

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`.

```
$ python caffe2npz.py [se-resnet50|se-resnet101|se-resnet152|se-resnext50|se-resnext101] <source>.caffemodel <target>.npz
```

Pretrained `.caffemodel` can be downloaded here: https://github.com/hujie-frank/SENet
