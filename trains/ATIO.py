"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from trains.singleTask import *

__all__ = ['ATIO']

# MODEL_MAP = {
#     'mult': MULT,
#     'tfn': TFN,
#     'lmf': LMF,
#     'mfn': MFN,
#     'ef_lstm': EF_LSTM,
#     'lf_dnn': LF_DNN,
#     'mtfn': MTFN,
#     'mlmf': MLMF,
#     'mlf_dnn': MLF_DNN,
# }

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'EF_LSTM': EF_LSTM,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.upper()](args)

