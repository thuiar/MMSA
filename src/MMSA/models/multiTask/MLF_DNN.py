import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MLF_DNN']

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class MLF_DNN(nn.Module):
    """
    late fusion using DNN
    """
    def __init__(self, args):
        super(MLF_DNN, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_out = args.text_out

        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, output_dim)

        # define the classify layer for text
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_out, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, output_dim)

        # define the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, output_dim)

        # define the classify layer for video
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, output_dim)


    def forward(self, text_x, audio_x, video_x):
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        # text
        x_t = self.post_text_dropout(text_h)
        x_t = F.relu(self.post_text_layer_1(x_t), inplace=True)
        x_t = F.relu(self.post_text_layer_2(x_t), inplace=True)
        output_text = self.post_text_layer_3(x_t)
        # audio
        x_a = self.post_audio_dropout(audio_h)
        x_a = F.relu(self.post_audio_layer_1(x_a), inplace=True)
        x_a = F.relu(self.post_audio_layer_2(x_a), inplace=True)
        output_audio = self.post_audio_layer_3(x_a)
        # video
        x_v = self.post_video_dropout(video_h)
        x_v = F.relu(self.post_video_layer_1(x_v), inplace=True)
        x_v = F.relu(self.post_video_layer_2(x_v), inplace=True)
        output_video = self.post_video_layer_3(x_v)

        # fusion
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output_fusion = self.post_fusion_layer_3(x)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res