"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from models.subNets.AlignNets import AlignSubNet
from models.singleTask import *
from models.multiTask import *

__all__ = ['AMIO']

MODEL_MAP = {
    'mult': MulT,
    'tfn': TFN,
    'lmf': LMF,
    'mfn': MFN,
    'ef_lstm': EF_LSTM,
    'lf_dnn': LF_DNN,
    'mtfn': MTFN,
    'mlmf': MLMF,
    'mlf_dnn': MLF_DNN,
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_align = args.need_align
        text_seq_len, _, _ = args.input_lens
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_align):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'input_lens' in args.keys():
                args.input_lens = self.alignNet.get_seq_len()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        if(self.need_align):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x)

