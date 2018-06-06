# Classification

## Performance

Single crop error rate.

| Model | Top 1 | Reference Top 1 |
|:-:|:-:|:-:|
| VGG16 | 29.0 % | 28.5 % [1] |
| ResNet50 | 24.8 % | 24.7 % [2] |
| ResNet101 | 23.6 % | 23.6 % [2] |
| ResNet152 | 23.2 % | 23.0 % [2] |

Ten crop error rate.

| Model | Top 1 | Reference Top 1 |
|:-:|:-:|:-:|
| VGG16 | 27.1 % |   |
| ResNet50 | 23.0 % | 22.9 % [2] |
| ResNet101 | 21.8 % | 21.8 % [2] |
| ResNet152 | 21.4 % | 21.4 % [2] |


The results can be reproduced by the following command.
The score is reported using a weight converted from a weight trained by Caffe.
These scores are obtained using OpenCV backend. If Pillow is used, scores would differ.

```
$ python eval_imagenet.py <path_to_val_dataset> [--model vgg16|resnet50|resnet101|resnet152] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>] [--crop center|10]
```


## How to prepare ImageNet Dataset

This instructions are based on the instruction found [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  $ mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  $ tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ..
  $ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  $ cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  $ mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  $ wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  $ mv ILSVRC2012_img_val.tar .. && cd ..
  ```


## References

1. Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition" ICLR 2015
2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition" CVPR 2016
