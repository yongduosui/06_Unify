GPU=$1

for i in 1e-2 1e-3 1e-4 1e-5 1e-6
do
    S1=$i
    S2=$i
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u main_gingat_imp.py \
    --net gat \
    --s1 $S1 \
    --s2 $S2 \
    --fix_epoch 200 \
    --mask_epoch 200
done