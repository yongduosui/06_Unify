CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp_seed.py \
--dataset cora --net gat \
--embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 \
--pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-3 --s2 1e-3