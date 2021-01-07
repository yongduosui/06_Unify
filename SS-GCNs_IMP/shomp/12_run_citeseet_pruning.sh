GPU=$1
DIM=512
adj=0.05
wei=0.1
s1=1e-5
s2=5e-4
for i in 0.1 0.19 0.271 0.3439 0.4095 0.4686 0.5217 0.5695 0.6126 0.6513
do
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u main_pruning_omp.py \
    --weight_dir ../GraphCL/cite_double_dgi.pkl \
    --dataset citeseer \
    --embedding-dim 3703 ${DIM} 6 \
    --lr 0.01 \
    --weight-decay 5e-4 \
    --pruning_percent_wei ${wei} \
    --pruning_percent_adj ${adj} \
    --total_epoch 500 \
    --s1 $s1 \
    --s2 $s2
done
