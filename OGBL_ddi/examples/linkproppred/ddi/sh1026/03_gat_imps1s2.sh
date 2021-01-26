S1=$1
S2=$2
GPU=$3
echo syd s1: $S1 s2: $S2
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_gingat_imp.py \
--net gat \
--s1 $S1 \
--s2 $S2 \
--fix_epoch 200 \
--mask_epoch 200
