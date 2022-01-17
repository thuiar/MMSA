[![](https://badgen.net/badge/license/MIT/blue)](#License) [![](https://badgen.net/badge/contact/THUIAR/purple)](https://thuiar.github.io/) ![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# MMSA

> A unified framwork for Multimodal Sentiment Analysis tasks.

**Note:** We strongly recommend browsing the overall structure of our code first. Feel free to contact us if you require any further information.

## Supported Models

|    Type     |                   Model Name                   |                                          From                                          |
| :---------: | :--------------------------------------------: | :------------------------------------------------------------------------------------: |
| Single-Task |    [EF_LSTM](models/singleTask/EF_LSTM.py)     |               [MultimodalDNN](https://github.com/rhoposit/MultimodalDNN)               |
| Single-Task |     [LF_DNN](models/singleTask/LF_DNN.py)      |                                           -                                            |
| Single-Task |        [TFN](models/singleTask/TFN.py)         |        [Tensor-Fusion-Network](https://github.com/A2Zadeh/TensorFusionNetwork)         |
| Single-Task |        [LMF](models/singleTask/LMF.py)         | [Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) |
| Single-Task |        [MFN](models/singleTask/MFN.py)         |               [Memory-Fusion-Network](https://github.com/pliang279/MFN)                |
| Single-Task |  [Graph-MFN](models/singleTask/Graph_MFN.py)   |            [Graph-Memory-Fusion-Network](https://github.com/pliang279/MFN)             |
| Single-Task | [MulT](models/singleTask/MulT.py)(without CTC) |      [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)      |
| Single-Task |   [BERT-MAG](models/singleTask/BERT_MAG.py)    |        [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer)        |
| Single-Task |        [MFM](models/singleTask/MFM.py)         |                                           -                                            |
| Single-Task |       [MISA](models/singleTask/MISA.py)        |                      [MISA](https://github.com/declare-lab/MISA)                       |
| Multi-Task  |     [MLF_DNN](models/multiTask/MLF_DNN.py)     |                         [MMSA](https://github.com/thuiar/MMSA)                         |
| Multi-Task  |        [MTFN](models/multiTask/MTFN.py)        |                         [MMSA](https://github.com/thuiar/MMSA)                         |
| Multi-Task  |        [MLMF](models/multiTask/MLMF.py)        |                         [MMSA](https://github.com/thuiar/MMSA)                         |
| Multi-Task  |     [SELF_MM](models/multiTask/SELF_MM.py)     |                      [Self-MM](https://github.com/thuiar/Self-MM)                      |

## Results

> Detailed results are shown in [results/result-stat.md](results/result-stat.md)

## Usage

### Clone codes

- Clone this repo and install requirements. Create virtual environments if needed.

```
git clone https://github.com/thuiar/MMSA
cd MMSA
# conda create -n mmsa python=3.6
pip install -r requirements.txt
```

### Datasets and pre-trained berts

Download dataset features and pre-trained berts from the following links.

- [Baidu Cloud Drive](https://pan.baidu.com/s/1oksuDEkkd3vGg2oBMBxiVw) with code: `ctgs`
- [Google Cloud Drive](https://drive.google.com/drive/folders/1E5kojBirtd5VbfHsFp6FYWkQunk73Nsv?usp=sharing)

For all features, you can use `SHA-1 Hash Value` to check the consistency.

> `MOSI/unaligned_50.pkl`: `5da0b8440fc5a7c3a457859af27458beb993e088`  
> `MOSI/aligned_50.pkl`: `5c62b896619a334a7104c8bef05d82b05272c71c`  
> `MOSEI/unaligned_50.pkl`: `db3e2cff4d706a88ee156981c2100975513d4610`  
> `MOSEI/aligned_50.pkl`: `ef49589349bc1c2bc252ccc0d4657a755c92a056`  
> `SIMS/unaligned_39.pkl`: `a00c73e92f66896403c09dbad63e242d5af756f8`

Due to the size limitations, the MOSEI features and SIMS raw videos are available in `Baidu Cloud Drive` only. All dataset features are organized as:

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

For MOSI and MOSEI, the pre-extracted text features are from BERT, different from the original glove features in the [CMU-Multimodal-SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).

For SIMS, if you want to extract features from raw videos, you need to install [Openface Toolkits](https://github.com/TadasBaltrusaitis/OpenFace/wiki) first, and then refer our codes in the `data/DataPre.py`.

```
python data/DataPre.py --data_dir [path_to_Dataset] --language ** --openface2Path  [path_to_FeatureExtraction]
```

For bert models, you also can download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert). And then, convert tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html)

Then, modify `config/config_*.py` to update dataset pathes.

### Run

```
python run.py --modelName *** --datasetName ***
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
