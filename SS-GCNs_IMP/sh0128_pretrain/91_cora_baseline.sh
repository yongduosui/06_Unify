GPU=$1
DIM=512
EPOCH=200

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_baseline.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--total_epoch ${EPOCH} \
--weight_dir ../GraphCL/cora_double_dgi.pkl