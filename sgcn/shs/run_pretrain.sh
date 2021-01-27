GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u pretrain.py \
--dataset pubmed \
--model gcn \
--epochs 2


