cd examples/pspnet
sed -e "s/data_dir=args.data_dir, split='val')/data_dir=args.data_dir, split='val').slice[:5]/" -i train_multi.py

$MPIEXEC $PYTHON train_multi.py --dataset ade20k --model pspnet_resnet50 --batch-size 1 --iteration 10
$MPIEXEC $PYTHON train_multi.py --dataset ade20k --model pspnet_resnet101 --batch-size 1 --iteration 10

