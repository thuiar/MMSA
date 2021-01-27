import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
from transformers import *

class getFeatures():
    def __init__(self, working_dir, openface2Path, pretrainedBertPath):
        self.data_dir = os.path.join(working_dir, 'Processed')
        self.label_path = os.path.join(working_dir, 'metadata/sentiment')
        # padding
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        # toolkits path
        self.openface2Path = openface2Path
        self.pretrainedBertPath = pretrainedBertPath

    def __read_hog(self, filename, batch_size=5000):
        """
        From: https://gist.github.com/btlorch/6d259bfe6b753a7a88490c0607f07ff8
        Read HoG features file created by OpenFace.
        For each frame, OpenFace extracts 12 * 12 * 31 HoG features, i.e., num_features = 4464. These features are stored in row-major order.
        :param filename: path to .hog file created by OpenFace
        :param batch_size: how many rows to read at a time
        :return: is_valid, hog_features
            is_valid: ndarray of shape [num_frames]
            hog_features: ndarray of shape [num_frames, num_features]
        """
        all_feature_vectors = []
        with open(filename, "rb") as f:
            num_cols, = struct.unpack("i", f.read(4))
            num_rows, = struct.unpack("i", f.read(4))
            num_channels, = struct.unpack("i", f.read(4))

            # The first four bytes encode a boolean value whether the frame is valid
            num_features = 1 + num_rows * num_cols * num_channels
            feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
            feature_vector = np.array(feature_vector).reshape((1, num_features))
            all_feature_vectors.append(feature_vector)

            # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
            num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
            # Read in batches of given batch_size
            num_floats_to_read = num_floats_per_feature_vector * batch_size
            # Multiply by 4 because of float32
            num_bytes_to_read = num_floats_to_read * 4

            while True:
                bytes = f.read(num_bytes_to_read)
                # For comparison how many bytes were actually read
                num_bytes_read = len(bytes)
                assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
                num_floats_read = num_bytes_read // 4
                assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
                num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

                feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
                # Convert to array
                feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
                # Discard the first three values in each row (num_cols, num_rows, num_channels)
                feature_vectors = feature_vectors[:, 3:]
                # Append to list of all feature vectors that have been read so far
                all_feature_vectors.append(feature_vectors)

                if num_bytes_read < num_bytes_to_read:
                    break

            # Concatenate batches
            all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

            # Split into is-valid and feature vectors
            is_valid = all_feature_vectors[:, 0]
            feature_vectors = all_feature_vectors[:, 1:]

            return is_valid, feature_vectors

    def getTextEmbedding(self, text):
        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        pretrained_weights = pretrainedBertPath
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        # add_special_tokens will add start and end token
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze().numpy()
    
    def __getAudioEmbedding(self, audio_path):
        y, sr = librosa.load(audio_path)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512 # hop_length smaller, seq_len larger
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)

        return np.concatenate([f0, mfcc, cqt], axis=-1) # (seq_len, 33)
    
    def __getVideoEmbedding(self, csv_path, pool_size=5):
        df = pd.read_csv(csv_path)

        features, local_features = [], []
        for i in range(len(df)):
            local_features.append(np.array(df.loc[i][df.columns[5:]]))
            if (i + 1) % pool_size == 0:
                features.append(np.array(local_features).mean(axis=0))
                local_features = []
        if len(local_features) != 0:
            features.append(np.array(local_features).mean(axis=0))
        return np.array(features)
    
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
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]
        # confirm length using (mean + std)
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence
    
    def handleImages(self):
        image_dirs = sorted(glob(os.path.join(self.data_dir, 'video/AlignedFaces', '*/*')))
        for image_dir in tqdm(image_dirs):
            output_dir = image_dir.replace('AlignedFaces', 'OpenFace2')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # cmd = '/home/****/ToolKits/OpenFace/build/bin/FeatureExtraction -fdir ' + image_dir + ' -out_dir ' + output_dir
            cmd = self.openface2Path + ' -fdir ' + image_dir + ' -out_dir ' + output_dir
            os.system(cmd)

    def results(self, output_dir):
        df_label_T = pd.read_csv(os.path.join(self.label_path, 'label_T.csv'))
        df_label_A = pd.read_csv(os.path.join(self.label_path, 'label_A.csv'))
        df_label_V = pd.read_csv(os.path.join(self.label_path, 'label_V.csv'))
        df_label_M = pd.read_csv(os.path.join(self.label_path, 'label_M.csv'))

        ids, rawText = [], []
        features_T, features_A, features_V = [], [], []
        label_T, label_A, label_V, label_M = [], [], [], []
        for i in tqdm(range(len(df_label_T))):
            video_id, clip_id = df_label_T.loc[i, ['video_id', 'clip_id']]
            clip_id = '%04d' % clip_id
            # text
            text = df_label_T.loc[i, 'text']
            embedding_T = self.__getTextEmbedding(text)
            features_T.append(embedding_T)
            # audio
            audio_path = os.path.join(self.data_dir, 'audio', video_id, clip_id + '.wav')
            embedding_A = self.__getAudioEmbedding(audio_path)
            features_A.append(embedding_A)
            # video
            csv_path = os.path.join(self.data_dir, 'video/OpenFace2', video_id, clip_id, clip_id+'.csv')
            embedding_V = self.__getVideoEmbedding(csv_path, pool_size=5)
            features_V.append(embedding_V)
            # labels
            label_T.append(df_label_T.loc[i, 'label'])
            label_A.append(df_label_A.loc[i, 'label'])
            label_V.append(df_label_V.loc[i, 'label'])
            label_M.append(df_label_M.loc[i, 'label'])
            # other
            ids.append(video_id + '_' + clip_id)
            rawText.append(text)
        # padding
        feature_T = self.__paddingSequence(features_T)
        feature_A = self.__paddingSequence(features_A)
        feature_V = self.__paddingSequence(features_V)
        # save
        results = {}
        results['feature_T'] = feature_T
        results['feature_A'] = feature_A
        results['feature_V'] = feature_V
        results['label_T'] = label_T
        results['label_A'] = label_A
        results['label_V'] = label_V
        results['label_M'] = label_M

        save_path = os.path.join(self.data_dir, output_dir, 'unaligned.pkl')
        with open(save_path, 'wb') as wf:
            plk.dump(results, wf, protocol = 4)
        print('Features are saved in %s!' %save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset/CH-SIMS',
                        help='path to CH-SIMS')
    parser.add_argument('--openface2Path', type=str, default="/home/iyuge2/ToolKits/OpenFace/build/bin/FeatureExtraction",
                        help='path to FeatureExtraction tool in openface2')
    parser.add_argument('--pretrainedBertPath', type=str, default="/home/sharing/disk3/pretrained_embedding/bert/chinese_L-12_H-768_A-12",
                        help='path to pretrained bert directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # data_dir = '/path/to/MSA-ZH'
    gf = getFeatures(args.data_dir, args.openface2Path, args.pretrainedBertPath)
    
    # gf.handleImages()

    # gf.results('features')
    gf.getTextEmbedding('我喜欢你')