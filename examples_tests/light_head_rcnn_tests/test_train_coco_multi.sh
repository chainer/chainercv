cd examples/light_head_rcnn
sed -e "s/split='train')/split='train').slice[:4]/" -i train_coco_multi.py

$MPIEXEC $PYTHON train_coco_multi.py --batchsize 4 --epoch 3 --step-epoch 1 2
