import argparse
import os
import pickle

import numpy as np
import torch
from models.subNets.BertTextEncoder import BertTextEncoder
from tqdm import tqdm


class TextPre(object):
    """A single set of features of data."""

    def __init__(self, args):
        self.device = torch.device('cuda:0')
        self.args = args
        self.loadTextMap = {
            'mosi': self.__load_data_mosi,
            'mosei': self.__load_data_mosei
        }
        self.bert = BertTextEncoder(language=args.language).to(self.device)
    
    def textConvertID(self, data, tokenizer):
        features = {}
        Input_ids, Input_mask, Segment_ids = [], [], []
        Raw_text, Visual, Audio = [], [], []
        Label, ids = [], []
        max_seq_length = self.args.max_seq_length
        for i in tqdm(range(len(data['raw_text']))):
            raw_text = data['raw_text'][i]
            visual = data['vision'][i]
            audio = data['audio'][i]
            tokens_a, inversions_a = tokenizer.tokenize(raw_text,invertable=True)
            
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length - 2]
                inversions_a = inversions_a[:max_seq_length - 2]
            
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))


            if self.args.aligned:
                text_len = min(len(raw_text.split()), max_seq_length)
                new_visual = [visual[len(visual) - text_len + inv_id] for inv_id in inversions_a]
                new_audio = [audio[len(audio) - text_len + inv_id] for inv_id in inversions_a]

                visual = np.array(new_visual)
                audio = np.array(new_audio)

                # add "start" and "end" for audio and vision
                audio_zero = np.zeros((1,audio.shape[1]))
                audio = np.concatenate((audio_zero,audio,audio_zero))

                visual_zero = np.zeros((1,visual.shape[1]))
                visual = np.concatenate((visual_zero,visual,visual_zero))

                audio_padding = np.zeros((max_seq_length - len(input_ids),audio.shape[1]))
                audio = np.concatenate((audio,audio_padding))

                video_padding = np.zeros((max_seq_length - len(input_ids),visual.shape[1]))
                visual = np.concatenate((visual,video_padding))

                assert audio.shape[0] == max_seq_length
                assert visual.shape[0] == max_seq_length

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label = float(data['labels'][i])

            Input_ids.append(input_ids)
            Visual.append(visual)
            Audio.append(audio)
            Input_mask.append(input_mask)
            Segment_ids.append(segment_ids)
            Label.append(label)
            Raw_text.append(raw_text)
            ids.append(data['id'][i])

        features['raw_text'] = np.array(Raw_text)
        features['audio'] = np.array(Audio)
        features['vision'] = np.array(Visual)
        features['labels'] = np.array(Label)
        features['id'] = np.array(ids)
        Input_ids = np.expand_dims(Input_ids, 1)
        Input_mask = np.expand_dims(Input_mask, 1)
        Segment_ids = np.expand_dims(Segment_ids, 1)
        text_bert = np.concatenate((Input_ids, Input_mask, Segment_ids), axis=1) 
        features['text_bert'] = text_bert
        features['text'] = self.__convertID2Vector(text_bert)
        return features
    
    def __convertID2Vector(self, ids, batch_size=64):
        results = []
        left = 0
        ids = torch.Tensor(ids)
        for left in tqdm(range(0, ids.size(0), batch_size)):
            right = min(left + batch_size, ids.size(0))
            c_ids = ids[left:right].to(self.device)
            c_vector = self.bert(c_ids).detach().cpu().numpy()
            results.append(c_vector)
        results = np.concatenate(results, axis=0)
        return results
    
    def __load_data_mosi(self):
        # get text data
        link = os.path.join(self.args.data_dir, 'Raw/Transcript/Segmented')
        text_data = {}
        for file in os.listdir(link):
            name = file.split('.')[0]
            for line in open(os.path.join(link, file), "r"):
                num_id, cur_t = line.split('_DELIM_')
                name_id = name + '_' + num_id.strip()
                text_data[name_id] = cur_t.strip()
        # get data
        def matchData(mode='train'):
            r_text = []
            for cur_id in data[mode]['id']:
                r_text.append(text_data[cur_id[0]])
            data[mode]['raw_text'] = r_text
        
        with open(os.path.join(self.args.data_dir, 'Processed/mosei_senti_data_noalign.pkl'), 'rb') as lf:
            data = pickle.load(lf)
        
        matchData(mode='train')
        matchData(mode='valid')
        matchData(mode='test')

        return data
    
    def __load_data_mosei(self):
        def convert0(s):
            if s == '0':
                return '0.0'
            return s
        # get text data
        link = os.path.join(self.args.data_dir, 'Raw/Transcript/Segmented')
        text_data = {}
        for file in os.listdir(link):
            name = file.split('.')[0]
            for line in open(os.path.join(link, file), "r"):
                items = line.split('___')
                name_id = items[0] + '_' + convert0(items[2]) + '_' + convert0(items[3])
                text_data[name_id.strip()] = items[-1].strip()
        # get data
        def matchData(mode='train'):
            r_text = []
            for cur_id in data[mode]['id']:
                name = '_'.join(cur_id)
                r_text.append(text_data[name])
            data[mode]['raw_text'] = r_text
        
        with open(os.path.join(self.args.data_dir, 'Processed/mosei_senti_data_noalign.pkl'), 'rb') as lf:
            data = pickle.load(lf)
        
        matchData(mode='train')
        matchData(mode='valid')
        matchData(mode='test')

        return data

    def run(self):
        data = self.loadTextMap[self.args.dataset_name]()

        train_list = data['train']
        valid_list = data['valid']
        test_list = data['test']

        tokenizer = self.bert.get_tokenizer()

        save_data = {}
        save_data['train'] = self.textConvertID(train_list, tokenizer)
        save_data['valid'] = self.textConvertID(valid_list, tokenizer)
        save_data['test'] = self.textConvertID(test_list, tokenizer)

        if self.args.aligned:
            saved_path = os.path.join(self.args.save_dir, 'aligned_' + str(self.args.max_seq_length) + '.pkl')
        else:
            saved_path = os.path.join(self.args.save_dir, 'unaligned_' + str(self.args.max_seq_length) + '.pkl')
        
        if not os.path.exists(os.path.dirname(saved_path)):
            os.makedirs(os.path.dirname(saved_path))

        with open(saved_path, 'wb') as file:
            pickle.dump(save_data, file, protocol=4)
            print('Save Successful!')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', type=str, default='mosei',
                        help='need aligned data (support mosi / mosei)')
    parser.add_argument('--language', type=str, default='cn',
                        help='data language')
    parser.add_argument('--aligned', type=bool, default=True,
                        help='need aligned data')
    parser.add_argument('--data_dir', type=str, default = '/home/sharing/disk3/dataset/multimodal-sentiment-dataset/CMU-MOSEI',
                        help='path to MOSI / MOSEI')
    parser.add_argument('--save_dir', type=str, default = '/home/sharing/disk3/dataset/multimodal-sentiment-dataset/ALL/mosei/raw',
                        help='path to saved directory')
    parser.add_argument('--max_seq_length', type=int, default = 50,
                        help='length')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tp = TextPre(args)
    tp.run()
    # tp.convertID2Vector()
