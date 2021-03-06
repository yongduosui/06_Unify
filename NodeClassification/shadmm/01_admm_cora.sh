GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_admm_omp_seed.py \
--dataset cora \
--embedding-dim 1433 512 7 \
--lr 0.008 \
--weight-decay 8e-5