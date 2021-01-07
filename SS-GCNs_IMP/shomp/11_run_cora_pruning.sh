GPU=$1
adj=0.1
wei=0.1
s1=1e-6
s2=5e-5
for i in 0.1 0.19 0.271 0.3439 0.4095 0.4686 0.5217 0.5695 0.6126 0.6513
do    
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u main_pruning_omp.py \
    --dataset cora \
    --embedding-dim 1433 16 7 \
    --lr 0.008 \
    --weight-decay 8e-5 \
    --pruning_percent_wei ${wei} \
    --pruning_percent_adj ${adj} \
    --total_epoch 500 \
    --s1 $s1 \
    --s2 $s2
done