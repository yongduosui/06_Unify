GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--dataset pubmed \
--embedding-dim 500 16 3 \
--lr 0.01 \
--weight-decay 5e-4