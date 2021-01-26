GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_baseline_gingat.py \
--net gat