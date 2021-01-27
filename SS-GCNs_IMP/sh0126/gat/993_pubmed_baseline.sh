GPU=$1
DIM=512
EPOCH=200
NET=gat

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_gingat_baseline.py \
--dataset pubmed \
--embedding-dim 500 ${DIM} 7 \
--lr 0.01 \
--weight-decay 5e-4 \
--total_epoch ${EPOCH} \
--net ${NET} \
--seed 123