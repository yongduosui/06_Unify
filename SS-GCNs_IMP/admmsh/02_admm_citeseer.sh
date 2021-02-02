GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_admm_omp.py \
--dataset citeseer \
--embedding-dim 3703 512 6 \
--lr 0.01 \
--weight-decay 5e-4