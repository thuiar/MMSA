![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# MMSA 
> Pytorch implementation for codes in CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality (ACL2020)

### Usage
---

#### Dataset
![Annotations](assets/Annotations.png)

- You can download CH-SIMS from the following links.
> md5: `6a92dccd83373b48ac83257bddab2538`

1. [Tsinghua Yun Disk](www.tsinghua.edu.cn)[code: `****`]
2. [Baidu Yun Disk](www.baidu.com)[code: `****`] 
3. [Google Drive](www.biying.com)[code: `****`]


#### Run the Code
- Clone this repo and install requirements.
```
git clone https://github.com/thuiar/MMSA  
cd MMSA
pip install -r requirements.txt
```

#### Data Preprocessing
> If you want to extract features from raw videos, you can use the following steps. Or you can directly use the feature data provided by us.

- **fetch audios and aligned faces (see `data/DataPre.py`)**
1. Install [ffmpeg toolkits](https://www.ffmpegtoolkit.com/)
```
sudo apt update
sudo apt install ffmpeg
```

2. Run `data/DataPre.py`
```
python data/DataPre.py --data_dir [path_to_CH-SIMS]
```

- **get features (see `data/getFeature.py`)**
1. Download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert).  
2. Convert Tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html)  
3. Install [Openface Toolkits](https://github.com/TadasBaltrusaitis/OpenFace/wiki)  
4. Run `data/getFeature.py`
```
python data/getFeature.py --data_dir [path_to_CH-SIMS] --openface2Path [path_to_FeatureExtraction] -- pretrainedBertPath [path_to_pretrained_bert_directory]
```
5. Then, you can see the preprocessed features in the `path/to/CH-SIMS/Processed/features/data.npz`

#### Run models
> you can set parameters in `config_run.py`. And then run `run.py`

```
python run.py --modelName *** --datasetName sims --tasks MTAV
```

### Citation

Please cite our paper if you find our work useful for your research:
> Wenmeng Yu, Hua Xu, etc. CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality