python calc_bn_statistics.py --gpu $1 --snapshot $2
python evaluate.py --gpu $1 --snapshot $2_infer
