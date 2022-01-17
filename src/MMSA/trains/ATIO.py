"""
AIO -- All Trains in One
"""
from .multiTask import *
from .singleTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mult': MULT,
            'bert_mag':BERT_MAG,
            'misa': MISA,
            'mfm': MFM,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
