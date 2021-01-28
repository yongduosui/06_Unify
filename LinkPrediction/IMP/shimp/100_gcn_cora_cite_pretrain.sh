DATASET=cora
S1=1e-4
S2=1e-4
MASK_EPOCH=200
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_imp_pretrain.py \
--dataset ${DATASET} \
--mask_epoch ${MASK_EPOCH} \
--fix_epoch ${FIX_EPOCH} \
--s1 ${S1} \
--s2 ${S2} \
--weight_dir ../../GraphCL/cora_double_dgi.pkl


DATASET=citeseer
S1=1e-4
S2=1e-4
MASK_EPOCH=200
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_imp_pretrain.py \
--dataset ${DATASET} \
--mask_epoch ${MASK_EPOCH} \
--fix_epoch ${FIX_EPOCH} \
--s1 ${S1} \
--s2 ${S2} \
--weight_dir ../../GraphCL/cite_double_dgi.pkl