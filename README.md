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

`nohup bash sh1026/01_gat_imp.sh 0 > 012601_gat_imp.log &`

`nohup bash sh1026/02_gat_imp.sh 0 > 012602_gat_imp.log &`



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

`cd OGBN_collab/examples/ogb/ogbl_collab`

`----------------------------running--------------------------------------------`

`nohup bash sh/00_collab_baseline.sh 0 > 012400_collab_baseline.log &`

`nohup bash sh/01_collab_imp.sh 0 > 012401_collab_imp_setting1.log &`

`nohup bash sh/02_collab_imp.sh 0 > 012402_collab_imp_setting2.log &`

`nohup bash sh/03_collab_imp.sh 0 > 012403_collab_imp_setting3.log &`

`nohup bash sh/04_collab_imp.sh 0 > 012404_collab_imp_setting4.log &`

`----------------------------TODO RP--------------------------------------------`

`nohup bash sh/99_collab_rp.sh 1 0 > 012401_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 2 0 > 012402_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 3 0 > 012403_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 4 0 > 012404_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 5 0 > 012405_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 6 0 > 012406_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 7 0 > 012407_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 8 0 > 012408_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 9 0 > 012409_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 10 0 > 012410_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 11 0 > 012411_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 12 0 > 012412_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 13 0 > 012413_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 14 0 > 012414_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 15 0 > 012415_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 16 0 > 012416_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 17 0 > 012417_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 18 0 > 012418_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 19 0 > 012419_collab_rp.log &`

`nohup bash sh/99_collab_rp.sh 20 0 > 012420_collab_rp.log &`