DATASET=citeseer
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_rp.py \
--net gat \
--dataset ${DATASET} \
--fix_epoch ${FIX_EPOCH}