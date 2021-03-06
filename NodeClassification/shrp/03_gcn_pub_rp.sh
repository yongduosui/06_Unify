CUDA_VISIBLE_DEVICES=$1 \
python -u main_pruning_random_seed.py --dataset pubmed ---embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200
