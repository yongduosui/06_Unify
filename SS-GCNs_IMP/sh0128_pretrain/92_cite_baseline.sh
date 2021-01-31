GPU=$1
DIM=512
EPOCH=200

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_baseline.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--total_epoch ${EPOCH} \
--weight_dir ../GraphCL/cite_double_dgi.pkl