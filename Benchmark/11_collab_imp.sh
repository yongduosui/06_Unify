GPU=$1
WEI=0.2
ADJ=0.05
seed0=41
EPOCH=500
dataset=OGBL-COLLAB
S1=5e-5
S2=5e-5
SAVE=IMP_setting1

echo syd setting1
python -u main_COLLAB_imp.py \
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


S1=1e-5
S2=1e-5
SAVE=IMP_setting2

echo syd setting2
python -u main_COLLAB_imp.py \
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


S1=1e-6
S2=1e-6
SAVE=IMP_setting3

echo syd setting3
python -u main_COLLAB_imp.py \
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