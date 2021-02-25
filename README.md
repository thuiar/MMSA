![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# MMSA 
> Pytorch implementation for codes in multimodal sentiment analysis.

- Update
1. Fix some bugs.
2. Add more models.
3. Add task scheduling mechanism.
4. Update regression and classification results on MOSI, MOSEI, and SIMS datasets.

- [MMSA](#mmsa)
  - [Support Models](#support-models)
  - [Results](#results)
  - [Usage](#usage)
    - [Download datasets](#download-datasets)
    - [Run the Code](#run-the-code)
    - [Data Preprocessing](#data-preprocessing)
    - [Run](#run)
  - [Paper](#paper)

## Support Models
In this framework, we support the following methods:

|     Type    |   Model Name      |     From                |
|:-----------:|:----------------:|:------------------------:|
| Single-Task |[EF_LSTM](models/singleTask/EF_LSTM.py)|[MultimodalDNN](https://github.com/rhoposit/MultimodalDNN)|
| Single-Task |[LF_DNN](models/singleTask/LF_DNN.py)|      -       |
| Single-Task |[TFN](models/singleTask/TFN.py)|[Tensor-Fusion-Network](https://github.com/A2Zadeh/TensorFusionNetwork)|
| Single-Task |[LMF](models/singleTask/LMF.py)| [Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)|
| Single-Task |[MFN](models/singleTask/MFN.py)|[Memory-Fusion-Network](https://github.com/pliang279/MFN)|
| Single-Task |[Graph-MFN](models/singleTask/Graph_MFN.py)|[Graph-Memory-Fusion-Network](https://github.com/pliang279/MFN)|
| Single-Task |[MulT](models/singleTask/MulT.py)(without CTC) |[Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)|
| Single-Task |[MISA](models/singleTask/MISA.py) |[MISA](https://github.com/declare-lab/MISA)|
| Multi-Task  |[MLF_DNN](models/multiTask/MLF_DNN.py)|      [MMSA](https://github.com/thuiar/MMSA)  |
| Multi-Task  |[MTFN](models/multiTask/MTFN.py)      |      [MMSA](https://github.com/thuiar/MMSA)  |
| Multi-Task  |[MLMF](models/multiTask/MLMF.py)      |      [MMSA](https://github.com/thuiar/MMSA)  |
| Multi-Task  |[SELF_MM](models/multiTask/SELF_MM.py)      |  [Self-MM](https://github.com/thuiar/Self-MM)  |

## Results
> Detailed results are shown in [results/result-stat.md](results/result-stat.md)

## Usage

### Download datasets

1. Download datasets from the following links.

- MOSI and MOSEI
> download from [CMU-MultimodalSDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)

- SIMS
> download from [Baidu Yun Disk](https://pan.baidu.com/s/1CmLdhYSVnNFAyA0DkR6tdA)[code: `ozo2`] or [Google Drive](https://drive.google.com/file/d/1z6snOkOoy100F33lzmHHB_DUGJ47DaQo/view?usp=sharing)

2. Preprocess features and save as a pickle file with the following structure and using `data/DataPre.py`.

```python
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train" 
    "test": {***}, # same as the "train"
}
```
3. Modify `config/config_*.py` to update dataset pathes.

### Run the Code
- Clone this repo and install requirements.
```
git clone https://github.com/thuiar/MMSA  
cd MMSA
pip install -r requirements.txt
```

### Data Preprocessing
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

### Run

```
python run.py
```

## Paper

- [CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality](https://www.aclweb.org/anthology/2020.acl-main.343/)
- [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis](https://arxiv.org/abs/2102.04830)

Please cite our paper if you find our work useful for your research:
```
@inproceedings{yu2020ch,
  title={CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality},
  author={Yu, Wenmeng and Xu, Hua and Meng, Fanyang and Zhu, Yilin and Ma, Yixiao and Wu, Jiele and Zou, Jiyun and Yang, Kaicheng},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3718--3727},
  year={2020}
}
```
```
@article{yu2021learning,
  title={Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis},
  author={Yu, Wenmeng and Xu, Hua and Yuan, Ziqi and Wu, Jiele},
  journal={arXiv preprint arXiv:2102.04830},
  year={2021}
}
```