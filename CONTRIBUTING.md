# Contributing to ChainerCV

:tada: Thank you for your interest in contributing to ChainerCV :tada:

We welcome any kind of contributions to ChainerCV!

Feel free to submit a PR or raise an issue even if the related tasks are not currently supported by ChainerCV.
ChainerCV is intended to include any codes that make computer vision research and development easier with Chainer.

When sending a PR to ChainerCV, please make sure the following:

+ Follow the coding guideline used in Chainer. This will be tested automatically by Travis CI.
+ Write unittests to verify your code. The tests follow the convention used in Chainer. 
+ Write documentations.
+ When adding a neural network architecture (e.g. Faster RCNN), make sure that the implementation achieves performance on par with the reported scores in the original paper.
Also, please write a summary on the behavioural differences between the original implementation if there is any.
For example, this summary can include a note on the difference between the hyperparamters used in the implementations.
+ Try to follow the coding conventions used in ChainerCV (e.g. variable names and directory structures). If you are adding a code without any related precedents, try to follow conventions used in Chainer, if any.
Please feel free to discuss on the conventions to use including those that are already implemented.
Also, [this issue](https://github.com/pfnet/chainercv/issues/159) is a good place to start knowing the coding conventions.
