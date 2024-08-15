# TAWFN

A Deep Learning Framework for Protein Function Prediction.


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

Data set can be downloaded from  [here](https://disk.pku.edu.cn/link/AA19000E6B9D98480EA943A777A9161347).

```
tar -zxvf dataset.tar.gz
```
The dataset related files will be under `data/dataset`
#### To test the model:
```
python test.py  --device 0
                --task bp 
                --batch_size 64 
                --AF2model False
                --model_cnn ./MCNN/model_bp.pt
                --model_gcn ./AGCN/model_bp.pt
```
