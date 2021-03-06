CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_rp.py --dataset pubmed --net gin --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

