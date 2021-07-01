# MobileNet

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert TensorFlow model
Convert TensorFlow's `*.ckpt` to `*.npz`.

```
$ python tfckpt2npz.py mobilenetv2 <source>.ckpt <target>.npz
```

The pretrained `.ckpt` for mobilenet can be downloaded from here.
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
