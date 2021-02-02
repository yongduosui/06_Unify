GPU=$1
SAVE=Baseline

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--learn_t \
--t 0.1 \
--epochs 200 \
--model_save_path ${SAVE} \
--seed 666

# --resume_dir CKPTs/debug/IMP2_fixed_ckpt.pth