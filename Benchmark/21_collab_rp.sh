GPU=$1
WEI=0.2
ADJ=0.05
seed0=41
EPOCH=500
dataset=OGBL-COLLAB
S1=1e-4
S2=1e-4
SAVE=RP_setting1

python -u main_COLLAB_rp.py \
--dataset $dataset \
--gpu_id $1 \
--seed $seed0 \
--config 'configs/COLLAB_edge_classification_GCN_40k.json' \
--s1 ${S1} \
--s2 ${S2} \
--epochs ${EPOCH} \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--model_save_path ${SAVE}