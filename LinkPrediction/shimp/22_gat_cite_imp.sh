CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp_seed.py --net gat --dataset citeseer --mask_epoch 200 --fix_epoch 200--s1 1e-4 --s2 1e-4