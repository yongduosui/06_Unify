GPU=$1
DIM=512
EPOCH=200
NET=gin

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_gingat_baseline.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--total_epoch ${EPOCH} \
--net ${NET}