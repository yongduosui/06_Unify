CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_imp_seed.py --dataset citeseer --mask_epoch 200 --fix_epoch 200 --s1 1e-3 --s2 1e-3