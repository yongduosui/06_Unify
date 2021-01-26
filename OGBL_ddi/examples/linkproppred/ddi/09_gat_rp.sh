GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} python -u main_gingat_rp.py --fix_epoch 200 --net gat
