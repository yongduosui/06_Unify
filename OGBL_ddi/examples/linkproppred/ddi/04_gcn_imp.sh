GPU=$1
S1=1e-6
S2=1e-3
echo syd: s1: $S1 s2: $S2
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--s1 $S1 \
--s2 $S2 \
--fix_epoch 200 \
--mask_epoch 200