GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python train.py \
--multi_gpus \
--phase train \
--data_dir ../data