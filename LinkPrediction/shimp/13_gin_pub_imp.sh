CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp_seed.py --net gin --dataset pubmed --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2