GPU=$1
for i in 10
do
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u main_admm_eval.py \
    --dataset cora \
    --embedding-dim 1433 512 7 \
    --lr 0.008 \
    --weight-decay 8e-5 \
    --index $i
done