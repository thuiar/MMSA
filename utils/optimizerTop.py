from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ['OptimizerTop']

class OptimizerTop():
    def __init__(self):
        self.optim_dict = {
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

    def __MULT(self, model, args):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return optimizer

    def __MFN(self, model, args):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return optimizer

    def __TFN(self, model, args):
        optimizer = optim.Adam(list(model.parameters())[2:], lr=args.learning_rate)
        return optimizer

    def __LMF(self, model, args):
        optimizer = optim.Adam([{"params": list(model.parameters())[:3], "lr": args.factor_lr},
                                {"params": list(model.parameters())[5:], "lr": args.learning_rate}],
                                weight_decay=args.weight_decay)
        return optimizer

    def __EF_LSTM(self, model, args):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return optimizer

    def __LF_DNN(self, model, args):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return optimizer
    
    def __MTFN(self, model, args):
        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": args.video_weight_decay},
                                {"params": list(model.parameters())[:2], "lr": 0.0}],
                                lr=args.learning_rate)
        return optimizer

    def __MLMF(self, model, args):
        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": args.video_weight_decay},
                                {"params": list(model.parameters())[:3], "lr": args.factor_lr},
                                {"params": list(model.parameters())[3:5], "lr": 0.0}],
                                lr=args.learning_rate)
        return optimizer

    def __MLF_DNN(self, model, args):
        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": args.video_weight_decay}],
                                lr=args.learning_rate)
        return optimizer

    def getOptim(self, model, args):
        return self.optim_dict[args.modelName.lower()](model, args)