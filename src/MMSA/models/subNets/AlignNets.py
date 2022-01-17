import torch
import torch.nn as nn

__all__ = ['AlignSubNet']

class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        From: https://github.com/yaohungt/Multimodal-Transformer
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        # return pseudo_aligned_out, (pred_output_position_inclu_blank)
        return pseudo_aligned_out
        
class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        mode: the way of aligning
            avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d']

        in_dim_t, in_dim_a, in_dim_v = args.feature_dims
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.mode = mode

        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
            'conv1d': self.__conv1d
        }

        if mode == 'conv1d':
            self.conv1d_T = nn.Conv1d(seq_len_t, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_A = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_V = nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ctc':
            self.ctc_t = CTCModule(in_dim_t, self.dst_len)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len)

    def get_seq_len(self):
        return self.dst_len
    
    def __ctc(self, text_x, audio_x, video_x):
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x

    def __avg_pool(self, text_x, audio_x, video_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        audio_x = align(audio_x)
        video_x = align(video_x)
        return text_x, audio_x, video_x
    
    def __conv1d(self, text_x, audio_x, video_x):
        text_x = self.conv1d_T(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.conv1d_A(text_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.conv1d_V(text_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x
 
    def forward(self, text_x, audio_x, video_x):
        # already aligned
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        return self.ALIGN_WAY[self.mode](text_x, audio_x, video_x)