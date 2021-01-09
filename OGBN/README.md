# 03_GCNTickets
## 1. Related Repo
1. [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
2. [OGB](https://ogb.stanford.edu/docs/home/)
3. [S3DIS](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)
## 2. Requirements
1. Enveronment:

`bash deepgcn_env_install_my.sh`

2. Dataset: 
S3DIS Dataset Download in Dropbox Dir:

`YD/indoor3d_sem_seg_hdf5_data.zip `

Then:

`mv indoor3d_sem_seg_hdf5_data.zip imp/data`

3. Command:

1221:

`nohup bash 99_imp_sen.sh 0,1 > 122101_imp_baseline.log &`

1222:

`cd imp/ogb/ogbg_mol`
`nohup bash 99_imp_mol.sh 0 > 122201_imp_mol.log &`

`cd imp/ogb/ogbg_ppa`
`nohup bash 99_imp_ppa.sh 0 > 122202_imp_ppa.log &`

`cd imp/ogb/ogbn_arxiv`
`nohup bash 99_imp_arxiv.sh 0 > 122203_imp_arxiv.log &`

`cd imp/ogb/ogbn_proteins`
`nohup bash 99_imp_proteins.sh 0 > 122204_imp_proteins.log &`

## 3. Experiments
### 3.1 Baseline
| Task | Dataset | Model | Baseline | Paper | MEM/G| Epoch | Time | GPU |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| 3D Points            | S3DIS    | ResGCN-28    | 52.69  | 52.11 | 40    | 100 | 2.5d | vita1(4) |
| Graph Classification | molhiv   | DyResGEN-7   | 0.7932 | 0.786 | 1.1   | 300 | 6h  07min | vita4 |
| Graph Classification | ppa      | ResGEN-28    | 0.7761 | 0.771 | 9.3   | 200 | 13h 34min | vita4 |
| Node Classification  | arxiv    | ResGEN-28    | 0.7146 | 0.719 | 20    | 500 | 1h  11min | titan |
| Node Classification  | proteins | DyResGEN-112 | 0.8566 | 0.858 | 23    | 1000| 33h | titan |


### 3.2 IMP

| Sparsity(0.8^n) | 3SDIS | molhiv | ppa  | arxiv | proteins | 
| :---:           | :---: | :---:  | :---:| :---: | :---:    |
|  0              | 50.41 |0.7649  |0.7704|0.7123 |0.8590 |
|  1              |       | 0.7899 |0.7720| 0.7106|0.8518 |
|  2              |       | 0.7827| 0.7670| 0.7144|0.8541|
|  3              |       | 0.7909|      | 0.7202| 0.8569|
|  4              |       | 0.7941|      | 0.7108| 0.8491|
|  5              |       | 0.7716|      | 0.7184|       |
|  6              |       | 0.7833|      | 0.7147|       |
|  7              |       | 0.7839|      | 0.7122|       |
|  8              |       | 0.7905|      | 0.7079|       |
|  9              |       | 0.7864|      | 0.7031|       |
|  10             |       | 0.7875|      | 0.7116|       |
|  11             |       | 0.7935|      | 0.7037|       |
|  12             |       | 0.7663|      | 0.6932|       |
|  13             |       | 0.7787|      | 0.6666|       |
|  14             |       | 0.7939|      | 0.6669|       |
|  15             |       | 0.7803|      | 0.6431|       |
|  16             |       | 0.7798|      | 0.5312|       |
|  17             |       | 0.7359|      | 0.4674|       |
|  18             |       | 0.7849|      | 0.4927|       |