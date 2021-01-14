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

`cd OGBN`

`bash deepgcn_env_install_my.sh`

`cd OGBN/unify/ogb/ogbn_arxiv`

`nohup bash sh/110_imp_arxiv.sh 0 > 011401_imp.log &`

`nohup bash sh/111_imp_arxiv.sh 0 > 011402_imp.log &`

`nohup bash sh/112_imp_arxiv.sh 0 > 011403_imp.log &`

`nohup bash sh/113_imp_arxiv.sh 0 > 011404_imp.log &`

`nohup bash sh/114_imp_arxiv.sh 0 > 011405_imp.log &`

`nohup bash sh/115_imp_arxiv.sh 0 > 011406_imp.log &`

`nohup bash sh/116_imp_arxiv.sh 0 > 011407_imp.log &`

`nohup bash sh/117_imp_arxiv.sh 0 > 011408_imp.log &`



### 2. OMP
