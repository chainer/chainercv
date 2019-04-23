cd examples/fcis
sed -e "s/split='train')/split='train').slice[:6]/" -i train_sbd_multi.py
sed -e "s/split='val')/split='val').slice[:6]/" -i train_sbd_multi.py

$MPIEXEC $PYTHON train_sbd_multi.py --batchsize 2 --epoch 2 --cooldown-epoch 1
