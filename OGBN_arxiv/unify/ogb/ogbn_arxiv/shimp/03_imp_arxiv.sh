GPU=$1
WEI=0.2
ADJ=0.05
S1=1e-4
S2=1e-4
SAVE=IMP

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--self_loop \
--learn_t \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--s1 ${S1} \
--s2 ${S2} \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epochs 200 \
--fix_epochs 500 \
--model_save_path ${SAVE}

# --resume_dir CKPTs/debug/IMP2_fixed_ckpt.pth