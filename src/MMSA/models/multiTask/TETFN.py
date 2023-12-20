"""
Paper: TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..subNets.transformers_encoder.transformer import TransformerEncoder
from ..subNets import BertTextEncoder

__all__ = ['TETFN']

class TETFN(nn.Module):
    def __init__(self, args):
        super(TETFN, self).__init__()
        # text subnets
        self.args = args
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        # audio-vision subnets
        text_in, audio_in, video_in = args.feature_dims

        self.audio_model = AuViSubNet(
            audio_in, 
            args.a_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        
        self.video_model = AuViSubNet(
            video_in, 
            args.v_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        
        self.proj_l = nn.Conv1d(text_in, args.dst_feature_dims, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        # fusion subnets
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')

        self.trans_a_with_v = TextEnhancedTransformer(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        
        self.trans_v_with_a = TextEnhancedTransformer(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
        
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(6 * args.dst_feature_dims, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.dst_feature_dims, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.dst_feature_dims, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.dst_feature_dims, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.args.nheads,
                                  layers=2,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.args.relu_dropout,
                                  res_dropout=self.args.res_dropout,
                                  embed_dropout=self.args.embed_dropout,
                                  attn_mask=True)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()

        text = self.text_model(text)

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        text = self.proj_l(text.transpose(1,2))
        proj_x_a = audio.permute(2, 0, 1)
        proj_x_v = video.permute(2, 0, 1)
        proj_x_l = text.permute(2, 0, 1)
        
        text_h = torch.max(proj_x_l, dim=0)[0]
        audio_h = torch.max(proj_x_a, dim=0)[0]
        video_h = torch.max(proj_x_v, dim=0)[0]
        
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]   # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_v, proj_x_a, proj_x_l)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_a, proj_x_v, proj_x_l)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # fusion
        fusion_h = torch.cat([last_h_l, last_h_a, last_h_v], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text_h)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio_h)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video_h)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class TextEnhancedTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout) -> None:
        super().__init__()

        self.lower_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=1,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        self.upper_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )
    
    def forward(self, query_m, key_m, text):
        c = self.lower_mha(query_m, text, text)
        return self.upper_mha(key_m, c, c)

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, conv1d_kernel_size, dst_feature_dims, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.conv = nn.Conv1d(hidden_size, dst_feature_dims, kernel_size=conv1d_kernel_size, bias=False)
        

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        h, _ = self.rnn(x)
        h = self.conv(h.transpose(1,2))
        return h
