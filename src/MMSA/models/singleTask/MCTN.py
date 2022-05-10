from ntpath import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence #
from collections import OrderedDict
from ..subNets import BertTextEncoder
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout, depth, bidirectional=True, lengths=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.bidirectional=bidirectional
        # self.fc0 = nn.linear(input_dim, hid_dim)
        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=depth, dropout=dropout, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, hid_dim, bias = False)

    def forward(self, x, lengths): 
        '''
        x : (batch_size, sequence_len, in_size)
        '''

        # lengths=lengths.to("cpu")
        # self.embedded = self.dropout(self.fc0(x)) # embedded = [src_len, batch_size, emb_dim]
        
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_state = [n_layers * num_directions, batch_size, hid_dim]
        # packed_sequence = [src_len, batch_size, input_dim]
        enc_output, enc_state = self.rnn(x) # if h_0 is not give, it will be set 0 acquiescently
        if self.bidirectional:
            h = self.dropout(torch.add(enc_output[:,:,:self.hid_dim],enc_output[:,:,self.hid_dim:]))
        else:
            h = self.dropout(enc_state[0].squeeze())
        join = h
        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer
        
        # enc_hidden [-2, :, : ] is the last of the forwards RNN 
        # enc_hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.add(enc_state[0][-1],enc_state[0][-2]))) ####
        
        return join, s


class Attention(nn.Module):
    def __init__(self, hid_dim, lengths=None):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, s, join):
        
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim ]
        
        batch_size = join.shape[1]
        src_len = join.shape[0]
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        join = join.transpose(0, 1)
        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, join), dim = 2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, lengths=None):
        super().__init__()
        self.lengths=lengths
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5): #######
        
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcin
        # g
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src, self.lengths)
                
        # first input to the decoder is the <sos> tokens
        dec_input = trg[0,:]
        
        for t in range(1, trg_len):
            
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = trg[t,:]
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return enc_output, outputs


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, dropout, depth, attention, bidirectional, lengths=None):
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.attention = attention
        self.hid_dim=hid_dim
        # self.embedding = nn.Embedding(hid_dim, hid_dim)
        self.rnn = nn.LSTM(output_dim+hid_dim, hid_dim, num_layers=depth, dropout=dropout, bidirectional = self.bidirectional)
        self.fc_out = nn.Linear(hid_dim + hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, s, join):
             
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        dec_input = dec_input.unsqueeze(1).transpose(0, 1) # dec_input = [batch_size, 1]
        
        # embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]
        
        # a = [batch_size, 1, src_len]  
        a = self.attention(s, join).unsqueeze(1)
        
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        join = join.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, join).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((dec_input, c), dim = 2)
            
        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_state = self.rnn(rnn_input) # s.unsqueeze(0)

        if self.bidirectional:
            dec_output = torch.add(dec_output[:,:,:self.hid_dim],dec_output[:,:,self.hid_dim:])
            h = torch.add(dec_state[0][-1],dec_state[0][-2])

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c), dim = 1))
        
        return pred, h.squeeze(0)


class Regression(nn.Module):
    def __init__(self, hid_dim, dropout, bidirectional=False, lengths=None):
        super().__init__()
        # self.l2_factor=l2_factor
        self.hid_dim=hid_dim
        self.bidirectional=bidirectional
        self.rnn=nn.LSTM(hid_dim, hid_dim, num_layers=1, dropout=dropout, bidirectional = self.bidirectional)
        self.linear=nn.Linear(hid_dim,1)
        self.linear2=nn.Linear(hid_dim,1)
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()

    def forward(self, x):
        x = x
        activations,_ = self.rnn(x)
        attention=self.tanh(self.linear(activations))
        attention=self.softmax(attention.squeeze(2))
        attention=attention.repeat(self.hid_dim,1,1).transpose(0,1).transpose(1,2)
        sent_representation = torch.mul(activations, attention)
        sent_representation = sent_representation.sum(1)
        
        regression_score = self.linear2(sent_representation) #kernel_regularizer=l2(l2_factor)
        return regression_score 


class MCTN(nn.Module):
    def __init__(self, config):#######################
        """Construct MultiMoldal InfoMax model.
        Args: 
            config (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        #input_dim = 50
        hid_dim = 32
        output_dim = 50
        depth = [1,1]
        dropout = 0
        self.attn1=Attention(hid_dim)
        self.attn2=Attention(hid_dim)
        self.encoder1=Encoder(768, hid_dim, dropout, depth[0])
        self.decoder1=Decoder(768, hid_dim, dropout, depth[1], self.attn1, bidirectional=True)
        self.encoder2=Encoder(hid_dim, hid_dim, dropout, depth[0])
        self.decoder2=Decoder(768, hid_dim, dropout, depth[1],self.attn2, bidirectional=True)
        self.seq2seq1=Seq2Seq(self.encoder1,self.decoder1,'cuda', lengths=None)
        self.seq2seq2=Seq2Seq(self.encoder2,self.decoder2,'cuda',lengths=None)
        self.regression=Regression(hid_dim, dropout, bidirectional=False)
        # self.join=None
        # self.use_bert = config.use_bert
        #if self.use_bert:
            # text subnets
        #    self.bertmodel = BertTextEncoder(use_finetune=config.use_finetune, transformers=config.transformers, pretrained=config.pretrained)
    
    def forward(self, text, audio, video, tar, lengths):
        # if self.use_bert:
        #     text = self.text_model(text)
        padding1 = (0,768-20)
        video=F.pad(video, padding1)
        padding2 = (0,768-5)
        audio=F.pad(audio, padding2)

        join, video_1 = self.seq2seq1(text, video, 0.5)
        _, text_1 = self.seq2seq1(video_1, text, 0.5)
        join, audio_1 = self.seq2seq2(join, audio, 0.5)
        tar_1 = self.regression(join)

        loss_v = nn.MSELoss()(video_1,video)
        loss_t = nn.MSELoss()(text_1,text)
        loss_a = nn.MSELoss()(audio_1,audio)
        loss_y = nn.L1Loss()(tar_1, tar)
        loss_all = 0.1*loss_v+0.1*loss_t+0.1*loss_a+1.0*loss_y
        return loss_all, tar_1
  

