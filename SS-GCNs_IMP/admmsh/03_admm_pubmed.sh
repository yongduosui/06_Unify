GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_admm_omp.py \
--dataset pubmed \
--embedding-dim 500 512 3 \
--lr 0.01 \
--weight-decay 5e-4