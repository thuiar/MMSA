"""
From: https://github.com/declare-lab/Multimodal-Infomax
Paper: Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis
"""

import torch

from ..subNets import BertTextEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

import time
import math 

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence #
from transformers import BertModel, BertConfig


__all__ = ['MMIM']


class RNNEncoder(nn.Module):
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
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # lengths = lengths.to(torch.int64)
        bs = x.size(0)

        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)
        batch_size = mu.size(0)

        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive,-1))

        # For Gaussian Distribution Estimation
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos':None, 'neg':None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y) 
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                a = 17.0795
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))

        return lld, sample_dict, H


class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce


class Fusion(nn.Module): #SubNet
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(Fusion, self).__init__() #SubNet
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3


class MMIM(nn.Module):
    def __init__(self, config):
        """Construct MultiMoldal InfoMax model.
        Args: 
            config (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()

        assert config.use_bert == True
        output_dim = config.num_classes if config.train_mode == "classification" else 1
        self.config = config
        self.aligned = config.need_data_aligned
        self.add_va = config.add_va
        config.d_tout = config.feature_dims[0]

        if config.use_bert:
            # text subnets
            self.bertmodel = BertTextEncoder(use_finetune=config.use_finetune, transformers=config.transformers, pretrained=config.pretrained) #######

        self.visual_enc = RNNEncoder(
            in_size = config.feature_dims[2],
            hidden_size = config.d_vh,
            out_size = config.d_vout,
            num_layers = config.n_layer,
            dropout = config.dropout_v if config.n_layer > 1 else 0.0,
            bidirectional = config.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = config.feature_dims[1],
            hidden_size = config.d_ah,
            out_size = config.d_aout,
            num_layers = config.n_layer,
            dropout = config.dropout_a if config.n_layer > 1 else 0.0,
            bidirectional = config.bidirectional
        )

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = config.d_tout,
            y_size = config.d_vout,
            mid_activation = config.mmilb_mid_activation,
            last_activation = config.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = config.d_tout,
            y_size = config.d_aout,
            mid_activation = config.mmilb_mid_activation,
            last_activation = config.mmilb_last_activation
        )

        if config.add_va:
            self.mi_va = MMILB(
                x_size = config.d_vout,
                y_size = config.d_aout,
                mid_activation = config.mmilb_mid_activation,
                last_activation = config.mmilb_last_activation
            )

        dim_sum = config.d_aout + config.d_vout + config.d_tout

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = config.d_tout, # to be predicted
            y_size = config.d_prjh,
            n_layers = config.cpc_layers,
            activation = config.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = config.d_vout,
            y_size = config.d_prjh,
            n_layers = config.cpc_layers,
            activation = config.cpc_activation
        )
        self.cpc_za = CPC(
            x_size = config.d_aout,
            y_size = config.d_prjh,
            n_layers = config.cpc_layers,
            activation = config.cpc_activation
        )

        # Trimodal Settings
        self.fusion_prj = Fusion(
            in_size = dim_sum,
            hidden_size = config.d_prjh,
            n_class = output_dim,
            dropout = config.dropout_prj
        )
            
    def forward(self, text, audio, vision, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.bertmodel(text) # (batch_size, seq_len, emb_size)
        
        text_h = enc_word[:,0,:] # (batch_size, emb_size)

        audio , audio_lengths = audio
        vision, vision_lengths = vision

        if self.aligned:
            mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
            text_lengths = mask_len.squeeze(1).int().detach().cpu()
            audio_h = self.acoustic_enc(audio, text_lengths)
            vision_h = self.visual_enc(vision, text_lengths)
        else:
            audio_h = self.acoustic_enc(audio, audio_lengths)
            vision_h = self.visual_enc(vision, vision_lengths)

        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text_h, y=vision_h, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text_h, y=audio_h, labels=y, mem=mem['ta'])

            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=vision_h, y=audio_h, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text_h, y=vision_h)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text_h, y=audio_h)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=vision_h, y=audio_h)


        # Linear proj and pred
        fusion, preds = self.fusion_prj(torch.cat([text_h, audio_h, vision_h], dim=1))

        nce_t = self.cpc_zt(text_h, fusion)
        nce_v = self.cpc_zv(vision_h, fusion)
        nce_a = self.cpc_za(audio_h, fusion)
        
        nce = nce_t + nce_v + nce_a

        pn_dic = {'tv':tv_pn, 'ta':ta_pn, 'va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': vision_h,
            'Feature_f': fusion,
            'lld': lld,
            'nce': nce,
            'pn_dic': pn_dic,
            'H': H,
            'M': preds
        }

        return res
