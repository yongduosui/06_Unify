IMP=$1
GPU=$2
echo pruning ${IMP}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train-auto-admm-tuneParameter.py \
--dataset pubmed \
--ADMM 4 \
--epochs 100 \
--model gcn \
--target_acc 0 \
--prune_ratio ${IMP} \
--count ${IMP}




