# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

https://github.com/lightaime/deep_gcns_torch

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Dim | Classes | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Node-level | cora    | 2708 |  13264  | 1433 | 7 | 81.03±0.64 | 236.10 |
| Node-level | citeseer| 3327 |  4732   | 3703 | 6 | 70.94±0.77 | 236.95 |
| Node-level | pubmed  |19717 | 108365  | 500  | 3 | 79.16±0.19 | 152.15 |
| Node-level | arxiv   | -    | -   | -   | -  | 79.16±0.19 | 152.15 |


### 2.2 Unify Pruning

`bash deepgcn_env_install_my.sh`

### 0. DDI

`cd OGBL_ddi`

`conda create -n ogb python=3.6`

`bash syd_install.sh`

`pip install dgl-cu101==0.4.2`

`cd examples/linkproppred/ddi`

`bash sh1026/00_gat_debug.sh 0`

`nohup bash 01_gat_baseline.sh 0 > 012600_gat_baseline.log &`

`nohup bash sh1026/01_gat_imp.sh 0 > 012601_gat_imp.log &`

`nohup bash sh1026/02_gat_imp.sh 0 > 012602_gat_imp.log &`

`nohup bash sh1026/03_gat_imps1s2.sh 1e-3 1e-4 0 > 012603_gat_imp.log &`

`nohup bash sh1026/03_gat_imps1s2.sh 1e-3 1e-5 0 > 012604_gat_imp.log &`

`nohup bash sh1026/03_gat_imps1s2.sh 1e-3 1e-6 0 > 012605_gat_imp.log &`

`nohup bash sh1026/03_gat_imps1s2.sh 1e-4 1e-5 0 > 012606_gat_imp.log &`

`nohup bash sh1026/03_gat_imps1s2.sh 1e-5 1e-5 0 > 012607_gat_imp.log &`

### 1. Proteins


`cd OGBN_proteins/unify/ogb/ogbn_proteins`

`----------------------------TODO RP--------------------------------------------`


`nohup bash sh/02_rp_proteins.sh 13 0 > 012413_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 14 0 > 012414_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 15 0 > 012415_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 16 0 > 012416_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 17 0 > 012417_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 18 0 > 012418_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 19 0 > 012419_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 20 0 > 012420_rp_proteins.log &`


### 2. Collab

`cd OGBL_collab_new/examples/ogb/ogbl_collab`

`---------------------------Resume-------------------------------------------`

`nohup bash 02_collab_imp_resume.sh 1 0 CKPTS/IMP_New_setting1/IMP1_train_ckpt.pth > 012901_collab_imp_setting1_resume.log & `

`nohup bash 02_collab_imp_resume.sh 2 0 CKPTS/IMP_New_setting1/IMP2_train_ckpt.pth > 012902_collab_imp_setting1_resume.log & `

`nohup bash 02_collab_imp_resume.sh 3 0 CKPTS/IMP_New_setting1/IMP3_train_ckpt.pth > 012903_collab_imp_setting1_resume.log & `

`nohup bash 02_collab_imp_resume.sh 4 0 CKPTS/IMP_New_setting1/IMP4_train_ckpt.pth > 012904_collab_imp_setting1_resume.log & `

`----------------------------RUNNING-------------------------------------------`



`nohup bash 01_collab_imp.sh 1 0 > 012701_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 2 0 > 012702_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 3 0 > 012703_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 4 0 > 012704_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 5 0 > 012705_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 6 0 > 012706_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 7 0 > 012707_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 8 0 > 012708_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 9 0 > 012709_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 10 0 > 012710_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 11 0 > 012711_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 12 0 > 012712_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 13 0 > 012713_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 14 0 > 012714_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 15 0 > 012715_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 16 0 > 012716_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 17 0 > 012717_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 18 0 > 012718_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 19 0 > 012719_collab_imp_setting1.log &`

`nohup bash 01_collab_imp.sh 20 0 > 012720_collab_imp_setting1.log &`


### 3. ADMM

`pip install tensorflow-gpu==1.13.1`

`cd sgcn`

`bash admm_pubmed.sh 1 0`

`nohup bash admm_pubmed.sh 1 0 > 012801_admm_pub01.log &`