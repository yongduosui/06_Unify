CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_imp_seed.py --dataset cora --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2