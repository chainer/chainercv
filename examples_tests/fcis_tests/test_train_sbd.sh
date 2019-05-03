cd examples/fcis
sed -e "s/split='train')/split='train').slice[:5]/" -i train_sbd.py
sed -e "s/split='val')/split='val').slice[:5]/" -i train_sbd.py

$PYTHON train_sbd.py --epoch 2 --cooldown-epoch 1 --gpu 0
