cd examples/fcis
sed -e "s/split='train')/split='train').slice[:4]/" -i train_coco_multi.py
sed -e "s/split='valminusminival')/split='valminusminival').slice[:2]/" -i train_coco_multi.py
sed -e "s/return_crowded=True, return_area=True)/return_crowded=True, return_area=True).slice[:6]/" -i train_coco_multi.py

$MPIEXEC $PYTHON train_coco_multi.py --batchsize 2 --epoch 2 --cooldown-epoch 1
