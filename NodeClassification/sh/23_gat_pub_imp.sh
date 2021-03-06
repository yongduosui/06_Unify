CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp_seed.py \
--dataset pubmed --net gat \
--embedding-dim 500 512 3 --lr 0.01 \
--weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2