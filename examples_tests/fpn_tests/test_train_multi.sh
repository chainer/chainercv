cd examples/fpn

$MPIEXEC $PYTHON train_multi.py --model faster_rcnn_fpn_resnet50 --batchsize 4 --iteration 9 --step 6 8
$MPIEXEC $PYTHON train_multi.py --model faster_rcnn_fpn_resnet101 --batchsize 4 --iteration 9 --step 6 8
