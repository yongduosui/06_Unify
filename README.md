# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Dim | Classes | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Node-level | cora    | 2708 |  13264  | 1433 | 7 | 81.03±0.64 | 236.10 |
| Node-level | citeseer| 3327 |  4732   | 3703 | 6 | 70.94±0.77 | 236.95 |
| Node-level | pubmed  |19717 | 108365  | 500  | 3 | 79.16±0.19 | 152.15 |
| Node-level | arxiv   | -    | -   | -   | -  | 79.16±0.19 | 152.15 |


### 2.2 Unify Pruning

### 1. IMP

`cd OGBN_proteins`

`bash deepgcn_env_install_my.sh`

`cd OGBN_proteins/unify/ogb/ogbn_proteins`




`nohup bash sh/01_imp_proteins.sh 0 > 012301_imp_proteins_resume.log &`

`nohup bash sh/02_imp_proteins.sh 0 > 012302_imp_proteins_resume.log &`

`nohup bash sh/03_imp_proteins.sh 0 > 012303_imp_proteins_resume.log &`

`nohup bash sh/04_imp_proteins.sh 0 > 012304_imp_proteins_resume.log &`

`nohup bash sh/06_imp_proteins.sh 0 > 012306_imp_proteins.log &`

`nohup bash sh/07_imp_proteins.sh 0 > 012307_imp_proteins.log &`

`nohup bash sh/08_imp_proteins.sh 0 > 012308_imp_proteins.log &`

`nohup bash sh/09_imp_proteins.sh 0 > 012309_imp_proteins.log &`

`nohup bash sh/10_imp_proteins.sh 0 > 012310_imp_proteins.log &`

`nohup bash sh/11_imp_proteins.sh 0 > 012311_imp_proteins.log &`

`nohup bash sh/12_imp_proteins.sh 0 > 012312_imp_proteins.log &`









### 2. OMP
