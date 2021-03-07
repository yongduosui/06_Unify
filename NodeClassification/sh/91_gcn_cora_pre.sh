CUDA_VISIBLE_DEVICES=$1 \
python -u main_pruning_imp_pretrain_seed.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one --weight_dir cora_double_dgi.pkl

