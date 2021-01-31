GPU=$1
DIM=512
EPOCH=200

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_baseline.py \
--dataset pubmed \
--embedding-dim 500 ${DIM} 3 \
--lr 0.01 \
--weight-decay 5e-4 \
--total_epoch ${EPOCH} \
--weight_dir ../GraphCL/pubmed_double_dgi.pkl