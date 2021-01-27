GPU=$1
# j=0
# for i in 5 9.75 14.26 18.55 22.62 26.49 30.17 33.66 36.98 40.13 43.12 45.96 48.67 51.23 53.67 55.99 58.19 60.28 62.26 64.15
# do
#     echo pruning $i
#     ((j++))
#     CUDA_VISIBLE_DEVICES=${GPU} \
#     python -u train-auto-admm-tuneParameter.py \
#     --dataset cora \
#     --ADMM 4 \
#     --epochs 100 \
#     --model gcn \
#     --target_acc 0 \
#     --prune_ratio $i \
#     --count $j
# done

i=48.67
echo pruning $i
((j++))
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train-auto-admm-tuneParameter.py \
--dataset cora \
--ADMM 4 \
--epochs 100 \
--model gcn \
--target_acc 0 \
--prune_ratio $i \
--count $j




