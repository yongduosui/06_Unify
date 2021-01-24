GPU=$1
WEI=0.2
ADJ=0.05
SAVE=RP2

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_rp.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--epochs 500 \
--model_save_path ${SAVE} \
--seed 666

# --resume_dir CKPTs/debug/IMP2_fixed_ckpt.pth