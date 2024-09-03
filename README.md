# TAWFN

A Deep Learning Framework for Protein Function Prediction.

Most of the codes in this study are obtained from  [HEAL](https://github.com/ZhonghuiGu/HEAL)  and  [MMSMAPlus](https://github.com/wzy-2020/MMSMAPlus). For more details one can check the original papers at:

[Wang Z, Deng Z, Zhang W, Lou Q, Choi KS, Wei Z, Wang L, Wu J. MMSMAPlus: a multi-view multi-scale multi-attention embedding model for protein function prediction. Brief Bioinform. 2023 Jul 20;24(4):bbad201.](https://doi.org/10.1093/bib/bbad201)

[Zhonghui Gu, Xiao Luo, Jiaxiao Chen, Minghua Deng, Luhua Lai, Hierarchical graph transformer with contrastive learning for protein function prediction, Bioinformatics, Volume 39, Issue 7, July 2023, btad410.](https://doi.org/10.1093/bioinformatics/btad410)

## Setup Environment

Clone the current repo

-   The code was developed and tested using python 3.7.
-   Clone the repository:  `git clone https://github.com/ss0830/TAWFN.git`
-   `conda env create -f environment.yml`.
-   Install PyTorch:  `conda install pytorch==1.7.0 cudatoolkit=10.2 -c pytorch`
-   Install PYG:  
```
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install *.whl
pip install torch_geometric==1.6.3
```
或者从[here](https://data.pyg.org/whl/)下载对应的文件再进行安装
##  Model testing

```
cd data
```

Data set can be downloaded from  [here](https://pan.baidu.com/s/1H8o-LvBVKQOjPG6hAnJtvQ?pwd=oax1).

```
tar -zxvf dataset.tar.gz
```
The dataset related files will be under `data/dataset`
#### To test the model:
```
python test.py  --device 0
                --task bp 
                --batch_size 64 
                --model_cnn ./MCNN/model_bp.pt
                --model_gcn ./AGCN/model_bp.pt
```
