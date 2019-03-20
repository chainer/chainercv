# MobileNet

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert Caffe model
Convert TensorFlow's `*.ckpt` to `*.npz`.

```
$ python caffe2npz.py mobilenetv2 <source>.ckpt <target>.npz
```

For the model architectures by shicai, the pretrained `.ckpt` for mobilenet can be downloaded from here.
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
