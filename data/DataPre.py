# coding: utf-8
import os
import shutil
import pickle
import librosa
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import torch
from transformers import BertTokenizer, BertModel

class dataPre():
    def __init__(self, args):
        self.working_dir = args.working_dir
        self.mtcnn = MTCNN(image_size=224, margin=0)
        # padding
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        # toolkits path
        self.openface2Path = args.openface2Path
        # bert
        tokenizer_class = BertTokenizer
        if args.language == 'cn':
            self.pretrainedBertPath = 'pretrained_model/bert_cn' 
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_cn')
        else:
            self.pretrainedBertPath = 'pretrained_model/bert_en' 
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_en', do_lower_case=True)

    def __getVideoEmbedding(self, video_path, tmp_dir, pool_size=5):
        faces_feature_dir = os.path.join(tmp_dir, 'Faces')
        os.mkdir(faces_feature_dir)
        cmd = self.openface2Path + ' -f ' + video_path + ' -out_dir ' + faces_feature_dir
        os.system(cmd)
        # read features
        df_path = glob(os.path.join(faces_feature_dir, '*.csv'))[0]
        df = pd.read_csv(df_path)
        features, local_features = [], []
        for i in range(len(df)):
            local_features.append(np.array(df.loc[i][df.columns[5:]]))
            if (i + 1) % pool_size == 0:
                features.append(np.array(local_features).mean(axis=0))
                local_features = []
        if len(local_features) != 0:
            features.append(np.array(local_features).mean(axis=0))
        return np.array(features)

    def __getAudioEmbedding(self, video_path, audio_path):
        # use ffmpeg to extract audio
        cmd = 'ffmpeg -i ' + video_path + ' -f wav -vn ' + \
                audio_path + ' -loglevel quiet'
        os.system(cmd)
        # get features
        y, sr = librosa.load(audio_path)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512 # hop_length smaller, seq_len larger
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)

        return np.concatenate([f0, mfcc, cqt], axis=-1)
    
    def __getTextEmbedding(self, text):
        # directory is fine
        tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)
        model = BertModel.from_pretrained(self.pretrainedBertPath)
        # add_special_tokens will add start and end token
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze().numpy()
    
    def __preTextforBert(self, text):
        tokens_a = self.tokenizer.tokenize(text,invertable=True)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        input_ids = np.expand_dims(input_ids, 1)
        input_mask = np.expand_dims(input_mask, 1)
        segment_ids = np.expand_dims(segment_ids, 1)

        text_bert = np.concatenate([input_ids, input_mask, segment_ids], axis=1)

        return text_bert

    def __padding(self, feature, MAX_LEN):
        """
        mode: 
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        assert self.padding_mode in ['zeros', 'normal']
        assert self.padding_location in ['front', 'back']

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]
        
        if self.padding_mode == "zeros":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif self.padding_mode == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN-length, feature.shape[1]))

        feature = np.concatenate([pad, feature], axis=0) if(self.padding_location == "front") else \
                  np.concatenate((feature, pad), axis=0)
        return feature

    def __paddingSequence(self, sequences):
        if len(sequences) == 0:
            return sequences
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]
        # confirm length using (mean + std)
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence
    
    def run(self):
        df_label = pd.read_csv(os.path.join(self.working_dir, 'label.csv'), dtype={'clip_id': str, 'video_id': str, 'text': str})
        # df_label = df_label[:10]

        output_path = os.path.join(self.working_dir, 'Processed/features.pkl')
        # load last point
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
            last_row_idx = np.sum([len(data[mode]['id']) for mode in ['train', 'valid', 'test']])
        else:
            data = {mode: {"id": [], 
                           "raw_text": [],
                           "audio": [],
                           "vision": [],
                           "text": [],
                           "text_bert": [],
                           "audio_lengths": [],
                           "vision_lengths": [],
                           "annotations": [],
                           "classification_labels": [], 
                           "regression_labels": []} for mode in ['train', 'valid', 'test']}
            last_row_idx = 0
        
        annotation_dict = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }

        isEnd = False

        try:
            tmp_dir = os.path.join(self.working_dir, 'Processed/tmp')
            for i in tqdm(range(last_row_idx, len(df_label))):
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                video_id, clip_id, text, label, annotation, mode, _ = df_label.loc[i]
                cur_id = video_id + '$_$' + clip_id
                # video
                video_path = os.path.join(self.working_dir, 'Raw', video_id, clip_id + '.mp4')
                embedding_V = self.__getVideoEmbedding(video_path, tmp_dir)
                seq_V = embedding_V.shape[0]
                # audio
                audio_path = os.path.join(tmp_dir, 'tmp.wav')
                embedding_A = self.__getAudioEmbedding(video_path, audio_path)
                seq_A = embedding_A.shape[0]
                # text
                embedding_T = self.__getTextEmbedding(text)
                text_bert = self.__preTextforBert(text)
                seq_T = embedding_T.shape[0]
                # save
                data[mode]['id'].append(cur_id)
                data[mode]['audio'].append(embedding_A)
                data[mode]['vision'].append(embedding_V)
                data[mode]['raw_text'].append(text)
                data[mode]['text'].append(embedding_T)
                data[mode]['text_bert'].append(text_bert)
                data[mode]['audio_lengths'].append(seq_A)
                data[mode]['vision_lengths'].append(seq_V)
                data[mode]['annotations'].append(annotation)
                data[mode]['classification_labels'].append(annotation_dict[annotation])
                data[mode]['regression_labels'].append(label)
                # clear tmp dir to save space
                shutil.rmtree(tmp_dir)
            isEnd = True
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            if isEnd:
                # padding
                for mode in ['train', 'valid', 'test']:
                    for item in ['audio', 'vision', 'text', 'text_bert']:
                        data[mode][item] = self.__paddingSequence(data[mode][item])
            with open(output_path, 'wb') as wf:
                pickle.dump(data, wf, protocol = 4)

            if isEnd:
                print("Completed!")
            print('Features are saved in %s!' %output_path)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset/StandardDatasets/MOSI',
                        help='path to datasets')
    parser.add_argument('--language', type=str, default="en",
                        help='en / cn')
    parser.add_argument('--openface2Path', type=str, default="/home/iyuge2/ToolKits/OpenFace/build/bin/FeatureExtraction",
                        help='path to FeatureExtraction tool in openface2')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dp = dataPre(args)
    dp.run()