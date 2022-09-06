from torch import nn


class LinearTrans(nn.Module):
    def __init__(self, args, modality='text'):
        super(LinearTrans, self).__init__()
        if modality == 'text':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[0]
        elif modality == 'audio':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[1]
        elif modality == 'vision':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[2]

        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.linear(x)

class Seq2Seq(nn.Module):
    def __init__(self, args, modality='text'):
        super(Seq2Seq, self).__init__()
        if modality == 'text':
            out_dim, in_dim = args.feature_dims[0], args.dst_feature_dim_nheads[0]*3
        elif modality == 'audio':
            out_dim, in_dim = args.feature_dims[1], args.dst_feature_dim_nheads[0]*3
        elif modality == 'vision':
            out_dim, in_dim = args.feature_dims[2], args.dst_feature_dim_nheads[0]*3

        self.decoder = nn.LSTM(in_dim, out_dim, num_layers=2, batch_first=True)

    def forward(self, x):
        return self.decoder(x)

MODULE_MAP = {
    'linear': LinearTrans,
}

class Generator(nn.Module):
    def __init__(self, args, modality='text'):
        super(Generator, self).__init__()

        select_model = MODULE_MAP[args.generatorModule]

        self.Model = select_model(args, modality=modality)

    def forward(self, x):
        return self.Model(x)
