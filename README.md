# SRIM-pytorch

Super-Resolution Implicit Model (SRIM) is a multi-modal image super-resoluition model. This repository contains two implementations of the model:

- One deep neural network featuring Residual-in-Residual Dense Block (RRDB)
- One legacy CNN based network derived from Caffe implementation

Code organization copied from [[BasicSR]](https://github.com/xinntao/BasicSR)

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))
- [PyTorch >= 1.1](https://pytorch.org)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [create_lmdb.py](codes/scripts/create_lmdb.py) for more details.

## Training and Testing
To train model with RRDB, please refer to [train_srim.sh](codes/train_srim.sh)
To train legacy Caffe model, please refer to [train_caffe.sh](codes/train_caffe.sh)


