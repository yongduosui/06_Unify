conda create -n deepgcn python=3.6
source activate deepgcn
# make sure pytorch version >=1.4.0
pip install torch==1.6.0 torchvision==0.7.0
pip install tensorboard

# command to install pytorch geometric, please refer to the official website for latest installation.
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu101
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric==1.4.3
pip install requests

# install useful modules
pip install tqdm
pip install ogb
