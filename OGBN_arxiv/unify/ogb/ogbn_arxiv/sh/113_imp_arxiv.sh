GPU=$1
WEI=0.2
ADJ=0.05
S1=5e-3
S2=1e-2
SAVE=IMP_setting4

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--s1 ${S1} \
--s2 ${S2} \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epochs 500 \
--fix_epochs 200 \
--model_save_path ${SAVE}

# --resume_dir CKPTs/debug/IMP2_fixed_ckpt.pth