import torch
from torch import nn

from ...subNets.BertTextEncoder import BertTextEncoder
from .alignment import Alignment
from .fusion import Fusion
from .generator import Generator


# CMD Loss
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=3):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        b = torch.max(x2, dim=0)[0]
        a = torch.min(x2, dim=0)[0]
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = (summed+1e-12)**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class RECLoss(nn.Module):
    def __init__(self, args):
        super(RECLoss, self).__init__()

        self.eps = torch.FloatTensor([1e-4]).to(args.device)
        self.args = args

        if args.recloss_type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif args.recloss_type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        elif args.recloss_type == 'cmd':
            self.loss = CMD()
        elif args.recloss_type == 'combine':
            self.loss = nn.SmoothL1Loss(reduction='sum')
            self.loss_cmd = CMD()

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        if self.args.recloss_type == 'combine' and self.args.weight_sim_loss!=0:
            loss += (self.args.weight_sim_loss * self.loss_cmd(pred*mask, target*mask) / (torch.sum(mask) + self.eps))
        return loss


class TFR_NET(nn.Module):
    def __init__(self, args):
        super(TFR_NET, self).__init__()
        self.args = args
        
        self.text_model = BertTextEncoder(use_finetune=args.use_bert_finetune)

        self.align_subnet = Alignment(args)

        if not args.without_generator:

            self.generator_t = Generator(args, modality='text')
            self.generator_a = Generator(args, modality='audio')
            self.generator_v = Generator(args, modality='vision')

            self.gen_loss = RECLoss(args)
   
        args.fusion_t_in = args.fusion_a_in = args.fusion_v_in = args.dst_feature_dim_nheads[0] * 3

        self.fusion_subnet = Fusion(args)
        

    def forward(self, text, audio, vision):
        text, text_m, missing_mask_t = text
        audio, audio_m, audio_mask, missing_mask_a = audio 
        vision, vision_m, vision_mask, missing_mask_v = vision

        text_mask = text[:,1,:]
        text_m = self.text_model(text_m)
        text = self.text_model(text)

        text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)

        if not self.args.without_generator:
        
            text_ = self.generator_t(text_h_g)
            audio_ = self.generator_a(audio_h_g)
            vision_ = self.generator_v(vision_h_g)

            text_gen_loss = self.gen_loss(text_, text, text_mask - missing_mask_t)
            audio_gen_loss = self.gen_loss(audio_, audio, audio_mask - missing_mask_a)
            vision_gen_loss = self.gen_loss(vision_, vision, vision_mask - missing_mask_v)

            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))

            return prediction, self.args.weight_gen_loss[0] * text_gen_loss + self.args.weight_gen_loss[1] * audio_gen_loss + self.args.weight_gen_loss[2] * vision_gen_loss
        else:
            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))
            return prediction, torch.Tensor([0]).to(self.args.device)
        