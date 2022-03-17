[![](https://badgen.net/badge/license/MIT/green)](#License)
[![](https://badgen.net/pypi/v/MMSA)](https://pypi.org/project/MMSA/) 
![](https://badgen.net/pypi/python/MMSA/)
[![](https://badgen.net/badge/contact/THUIAR/purple)](https://thuiar.github.io/)

# MMSA

MMSA is a unified framework for Multimodal Sentiment Analysis.

## Features

- Train, test and compare multiple MSA models in a unified framework.
- Supports [14]() MSA models, including recent works.
- Supports 3 MSA datasets: [MOSI](https://ieeexplore.ieee.org/abstract/document/7742221), [MOSEI](https://aclanthology.org/P18-1208.pdf), and [CH-SIMS](https://aclanthology.org/2020.acl-main.343/).
- Easy to use, provides Python APIs and commandline tools.
- Experiment with fully customized multimodal features extracted by [MMSA-FET](https://github.com/thuiar/MMSA-FET) toolkit.

## Get Started

> **Note:** From version 2.0, we packaged the project and uploaded it to PyPI in the hope of making it easier to use. If you don't like the new structure, you can always switch back to `v_1.0` branch. 

### 1. Use Python API

- Run `pip install MMSA` in your python virtual environment.
- Import and use in any python file:

  ```python
  from MMSA import MMSA_run

  # run LMF on MOSI with default hyper parameters
  MMSA_run('lmf', 'mosi', seeds=[1111, 1112, 1113], gpu_ids=[0])

  # tune TFN on MOSEI with default hyper parameter range
  MMSA_run('tfn', 'mosei', seeds=[1111], gpu_ids=[1])

  ```

- For more detailed usage, please refer to [this page]().

### 2. Use Commandline Tool

- Run `pip install MMSA` in your python virtual environment.
- Use from command line:

  ```bash
  # show usage
  $ python -m MMSA -h

  # train & test LMF on MOSI with default parameters
  $ python -m MMSA
  ```

- For more detailed usage, please refer to [this page]().

### 3. Clone & Edit the Code

- Clone this repo and install requirements.
  ```bash
  $ git clone https://github.com/thuiar/MMSA
  ```
- Edit the codes to your needs. Add a new model or alter the configs, etc.
- After editing, run the following commands:
  ```bash
  $ cd MMSA-master # make sure you're in the top directory
  $ pip install .
  ```
- Then run the code like above sections.
- To further change the code, you need to re-install the package:
  ```bash
  $ pip uninstall MMSA
  $ pip install .
  ```
- If you'd rather run the code without installation(like in v_1.0), please refer to [this page]().

## Datasets

MMSA currently supports MOSI, MOSEI, and CH-SIMS dataset. Use the following links to download raw videos, feature files and label files. You don't need to download raw videos if you're not planning to run end-to-end tasks. 

- All files: [BaiduYun Disk](https://pan.baidu.com/s/1F2CgPCeG4eI6nmrRwp4ESA?pwd=7ppk)
- Feature and label files only: [Google Drive](https://drive.google.com/drive/folders/1E5kojBirtd5VbfHsFp6FYWkQunk73Nsv?usp=sharing)

SHA-256 for feature files:

```text
`MOSI/Processed/unaligned_50.pkl`:  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`
`MOSI/Processed/aligned_50.pkl`:    `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`
`MOSEI/Processed/unaligned_50.pkl`: `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`
`MOSEI/Processed/aligned_50.pkl`:   `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`
`SIMS/Processed/unaligned_39.pkl`:  `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`
```

MMSA uses feature files that are organized as follows:

```python
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```

> **Note:** For MOSI and MOSEI, the pre-extracted text features are from BERT, different from the original glove features in the [CMU-Multimodal-SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).

> **Note:** If you wish to extract customized multimodal features, please try out our [MMSA-FET](https://github.com/thuiar/MMSA-FET)


## Supported MSA Models

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

> Baseline results are reported in [results/result-stat.md](results/result-stat.md)

## Paper

- [CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality](https://www.aclweb.org/anthology/2020.acl-main.343/)
- [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis](https://arxiv.org/abs/2102.04830)
- [M-SENA: An Integrated Platform for Multimodal Sentiment Analysis]()

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
@inproceedings{yu2021learning,
  title={Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis},
  author={Yu, Wenmeng and Xu, Hua and Yuan, Ziqi and Wu, Jiele},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={10790--10797},
  year={2021}
}
```
