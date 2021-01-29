DATASET=citeseer
FIX_EPOCH=5
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_rp.py \
--dataset ${DATASET} \
--fix_epoch ${FIX_EPOCH}