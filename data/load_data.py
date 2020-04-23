import os
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']

class MMDataset(Dataset):
    def __init__(self, args, index, mode='train'):
        self.mode = mode
        self.index = index
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'sims': self.__init_msaZH
        }
        DATA_MAP[args.datasetName](args)

    def __init_mosi(self, args):
        with open(args.datapath, 'rb') as f:
            data = pickle.load(f)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.label = {
            'M': data[self.mode]['labels'].astype(np.float32)
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['vision']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max==0] = 1
            self.__normalize()
    
    def __init_msaZH(self, args):
        data = np.load(args.datapath)
        self.vision = data['feature_V'][self.index[self.mode]]
        self.audio = data['feature_A'][self.index[self.mode]]
        self.text = data['feature_T'][self.index[self.mode]]
        self.label = {
            'M': data['label_M'][self.index[self.mode]], 
            'T': data['label_T'][self.index[self.mode]], 
            'A': data['label_A'][self.index[self.mode]], 
            'V': data['label_V'][self.index[self.mode]]
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['feature_V'][self.index['train']]), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max==0] = 1
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        # return len(self.labels)
        return len(self.index[self.mode])

    def get_seq_len(self):
        return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.label.items()}
        } 
        return sample

def MMDataLoader(args):
    # if 'is_ten_fold' in args.keys() and args.is_ten_fold:
    #     # 10 fold cross-validation
    #     k_fold_value = args.cur_time - 1
    #     test_index = [i for i in range(k_fold_value, args.nsamples, 10)]
    #     not_test_index = [i for i in range(args.nsamples) if i not in test_index]

    #     val_index = [not_test_index[i] for i in range(0, len(not_test_index), 10)]
    #     train_index = [i for i in range(args.nsamples) if i not in test_index and i not in val_index]
    # else:
    # fixed split
    test_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'test_index.csv'))).reshape(-1)
    train_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'train_index.csv'))).reshape(-1)
    val_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'val_index.csv'))).reshape(-1)

    print('Train Samples Num: {0}'.format(len(train_index)))
    print('Valid Samples Num: {0}'.format(len(val_index)))
    print('Test Samples Num: {0}'.format(len(test_index)))

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }
    datasets = {
        'train': MMDataset(args, index=index, mode='train'),
        'valid': MMDataset(args, index=index, mode='valid'),
        'test': MMDataset(args, index=index, mode='test')
    }

    if 'input_lens' in args.keys():
        args.input_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader