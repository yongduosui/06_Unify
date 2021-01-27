GPU=$1
for i in {1..20}
do
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u train.py \
    --dataset cora \
    --epochs 200 \
    --model gcn \
    --adj_index $i
done