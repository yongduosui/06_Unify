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

### 1. Proteins


`cd OGBN_proteins/unify/ogb/ogbn_proteins`


`----------------------------running--------------------------------------------`

`nohup bash sh/01_imp_proteins.sh 0 > 012301_imp_proteins_resume.log &`

`nohup bash sh/02_imp_proteins.sh 0 > 012302_imp_proteins_resume.log &`

`nohup bash sh/03_imp_proteins.sh 0 > 012303_imp_proteins_resume.log &`

`nohup bash sh/04_imp_proteins.sh 0 > 012304_imp_proteins_resume.log &`

`nohup bash sh/06_imp_proteins.sh 0 > 012306_imp_proteins.log &`

`nohup bash sh/07_imp_proteins.sh 0 > 012307_imp_proteins.log &`

`nohup bash sh/08_imp_proteins.sh 0 > 012308_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 9 0 > 012309_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 10 0 > 012310_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 11 0 > 012311_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 12 0 > 012312_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 13 0 > 012313_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 14 0 > 012314_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 15 0 > 012315_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 16 0 > 012316_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 17 0 > 012317_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 18 0 > 012318_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 19 0 > 012319_imp_proteins.log &`

`nohup bash sh/01_imp_proteins.sh 20 0 > 012320_imp_proteins.log &`

`----------------------------TODO RP--------------------------------------------`

`nohup bash sh/02_rp_proteins.sh 1 0 > 012401_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 2 0 > 012402_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 3 0 > 012403_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 4 0 > 012404_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 5 0 > 012405_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 6 0 > 012406_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 7 0 > 012407_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 8 0 > 012408_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 9 0 > 012409_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 10 0 > 012410_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 11 0 > 012411_rp_proteins.log &`

`nohup bash sh/02_rp_proteins.sh 12 0 > 012412_rp_proteins.log &`

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