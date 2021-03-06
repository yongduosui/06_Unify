CUDA_VISIBLE_DEVICES=$1 \
python -u main_pruning_imp_seed.py \
--dataset citeseer \
--embedding-dim 3703 512 6 \
--lr 0.01 --weight-decay 5e-4 \
--pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one