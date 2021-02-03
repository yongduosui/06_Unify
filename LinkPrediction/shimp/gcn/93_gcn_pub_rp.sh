DATASET=pubmed
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_rp.py \
--dataset ${DATASET} \
--fix_epoch ${FIX_EPOCH}