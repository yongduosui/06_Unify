GPU=$1
WEI=0.2
ADJ=0.05
S1=1e-7
S2=5e-7
SAVE=IMP_setting2

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