# ResNet

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`.

```
$ python caffe2npz.py [resnet50|resnet101|resnet152] <source>.caffemodel <target>.npz
```

For the model architectures by Kaiming He, the pretrained `.caffemodel` for ResNet can be downloaded from here.
https://github.com/KaimingHe/deep-residual-networks
