GPU=$1
seed0=41
dataset=OGBL-COLLAB

python -u main_COLLAB_edge_classification.py \
--dataset $dataset \
--gpu_id $1 \
--seed $seed0 \
--config 'configs/COLLAB_edge_classification_GCN_40k.json'