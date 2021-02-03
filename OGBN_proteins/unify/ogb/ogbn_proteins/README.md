# UGS Node Classification on Ogbn-Proteins
## 1. Requirements

```
bash deepgcn_env_install.sh
```


## 2. Training & Evaluation

### UGS and RP 

```
python -u main_imp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--s1 1e-1 \
--s2 1e-3 \
--epochs 100 \
--iteration 10 \
--model_save_path IMP \
--imp_num 1


python -u main_rp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--epochs 100 \
--iteration 10 \
--model_save_path RP \
--imp_num 1

```


