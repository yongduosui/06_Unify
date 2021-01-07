GPU=$1
PER=0.1
SEED=39
python -u execute.py \
--dataset cora \
--aug_type subgraph \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cora_best_dgi.pkl \
--gpu ${GPU}
