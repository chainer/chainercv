# Classification

## Performance

| Model | Top 1 Error | Original Top 1 Error |
|:-:|:-:|:-:|
| VGG16 | 27.06 % | 27.0 % [1] |

The results can be reproduced by the following command

```
$ python eval_imagenet.py <path_to_val_dataset> [--model vgg16] [--pretrained_model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
```


## How to prepare ImageNet Dataset

This instructions are copied from ImageNet training for Torch.

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

## Note on implementations

#### VGG16

In the original paper, fully connected layers are used as convolutional layers, and the final output is the spatial average of the outputs of the convolutions.
Our implementation averages predictions from ten-cropped patches.


## References

1. Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition" ICLR 2015
