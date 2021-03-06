CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_rp_seed.py --dataset citeseer --net gin --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

