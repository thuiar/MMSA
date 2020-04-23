import torch.nn as nn

__all__ = ['LossTop']

class LossTop():
    def __init__(self, args):
        self.loss_dict = {
            "mult": self.__MULT,
            "mfn": self.__MFN,
            "tfn": self.__TFN,
            "lmf": self.__LMF,
            "ef_lstm": self.__EF_LSTM,
            "lf_dnn": self.__LF_DNN,
            "mtfn": self.__MTFN,
            "mlmf": self.__MLMF,
            "mlf_dnn": self.__MLF_DNN
        }
        self.args = args

    def __MULT(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss

    def __MFN(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss

    def __TFN(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss

    def __LMF(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss
        
    def __EF_LSTM(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss

    def __LF_DNN(self, y_pred, y_true):
        loss = nn.L1Loss()(y_pred['M'], y_true['M'])
        return loss
    
    def __MTFN(self, y_pred, y_true):
        criterion = nn.L1Loss()
        loss = 0.0
        for m in self.args.modality:
            loss += eval('self.args.'+m) * criterion(y_pred[m], y_true[m])
        return loss

    def __MLMF(self, y_pred, y_true):
        criterion = nn.L1Loss()
        loss = 0.0
        for m in self.args.modality:
            loss += eval('self.args.'+m) * criterion(y_pred[m], y_true[m])
        return loss

    def __MLF_DNN(self, y_pred, y_true):
        criterion = nn.L1Loss()
        loss = 0.0
        for m in self.args.modality:
            loss += eval('self.args.'+m) * criterion(y_pred[m], y_true[m])
        return loss

    def getLoss(self):
        return self.loss_dict[self.args.modelName.lower()]