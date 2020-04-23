![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# MMSA 
> Pytorch implementation for codes in CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality (ACL2020)

### Support Models in This Repo.
#### baselines
- [EF-LSTM](models/baselines/MDNN.py)
- [LF-DNN](models/baselines/MDNN.py)
- [TFN](models/baselines/TFN.py)
- [LMF](models/baselines/LMF.py)
- [MFN](models/baselines/MFN.py)
- [MulT](models/baselines/MulT.py)

#### Our
- [MLF_DNN.py](models/our/MLF_DNN.py)
- [MTFN.py](models/our/MTFN.py)
- [MLMF.py](models/our/MLMF.py)
- [MMFN.py](models/our/MMFN.py)
- [MMulT.py](models/our/MMulT.py)


### File Tree
> Our code is organized with "Dataset+Model".
```
MMSA
├── asserts                         ## images folder
├── config                          ## parameters
│   ├── config_debug.py             ## adjust parameters with grid search
│   ├── config_run.py               ## config running parameters
├── data
│   ├── getFeature.py               ## get basic features from raw data
│   ├── __init__.py
│   ├── load_data.py                ## load data
├── models                          
│   ├── AIO.py                      ## top model
│   ├── baselines                   ## baselines
│   ├── __init__.py
│   ├── modules                     ## other middle modules
│   ├── our                         ## our methods
├── README.md
├── results                         ## results dir
├── run.py                          ## run
├── utils                           
│   ├── __init__.py
│   ├── log.py                      
│   ├── lossTop.py                 ## loss
│   ├── metricsTop.py              ## metrics
│   ├── optimizerTop.py            ## optimizer
```

### Usage
---
#### Prerequisites
- python3.6 or python3.7
- pytorch >= 1.2

#### Dataset
![Annotations](assets/Annotations.png)
- download CH-SIMS from [Baidu Yun Disk](www.baidu.com)(code: `****`) or [Microsoft Drive](www.biying.com).
- data preprocessing
> We provide raw data and features extracted by functions in `data/getFeature.py`.

#### Run the codes
- Clone this repo and install requirements.
```
git clone https://github.com/thuiar/MMSA  
cd MMSA
pip install -r requirements.txt
```

- Check Parameters
> 1. Check the hyperparameters in `run.py`.  
> 2. Check the hyperparameters in `config_run.py` (or `config_debug.py`). Config the parameters related to datasets or models.  

- Run
> python run.py

### Citation

Please cite our paper if you find our work useful for your research:
> Wenmeng Yu, Hua Xu, etc. CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality