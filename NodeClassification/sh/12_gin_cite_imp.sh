CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp_seed.py \
--dataset citeseer --net gin \
--embedding-dim 3703 512 6 --lr 0.01 \
--weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-5 --s2 1e-5