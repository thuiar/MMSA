'''
* @name: almt.py
* @description: Implementation of ALMT (https://github.com/Haoyu-ha/ALMT)
'''

import torch
from torch import nn, einsum
from ..subNets import BertTextEncoder
from einops import rearrange, repeat


class ALMT(nn.Module):
    def __init__(self, args):
        super(ALMT, self).__init__()
        if args.use_bert:
            self.bertmodel = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        self.use_bert = args.use_bert

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.orig_length_l, self.orig_length_a, self.orig_length_v = args.feature_length
        self.dst_embedding_d_l = self.dst_embedding_d_a = self.dst_embedding_d_v = args.dst_feature_dims
        self.dst_embedding_hidden_d_l = self.dst_embedding_hidden_d_a = self.dst_embedding_hidden_d_v = args.dst_feature_hidden_dims
        self.dst_embedding_length_l = self.dst_embedding_length_a = self.dst_embedding_length_v = args.dst_embedding_length
        self.embedding_depth_l, self.embedding_depth_a, self.embedding_depth_v = args.embedding_depth
        self.embedding_heads_l, self.embedding_heads_a, self.embedding_heads_v = args.embedding_heads

        self.l_encoder_hidden_d = self.h_hyper_d = args.dst_feature_dims
        self.l_encoder_heads = args.l_encoder_heads
        self.h_hyper_length = args.dst_embedding_length
        
        self.h_hyper = nn.Parameter(torch.ones(1, self.h_hyper_length, self.h_hyper_d))
        self.AHL_depth = args.AHL_depth
        self.h_hyper_layer_d = args.dst_feature_dims
        self.h_hyper_layer_heads = args.h_hyper_layer_heads

        self.fusion_d = self.h_hyper_d
        self.fusion_hidden_d = args.fusion_hidden_d
        self.fusion_heads = args.fusion_heads
        self.fusion_layer_depth = args.fusion_layer_depth


        # Embedding
        self.embedding_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.dst_embedding_d_l),
            Transformer(num_frames=self.orig_length_l, 
                        save_hidden=False, 
                        token_len=self.dst_embedding_length_l, 
                        dim=self.dst_embedding_d_l, 
                        depth=self.embedding_depth_l, 
                        heads=self.embedding_heads_l, 
                        mlp_dim=self.dst_embedding_hidden_d_l)
        )

        self.embedding_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.dst_embedding_d_a),
            Transformer(num_frames=self.orig_length_a, 
                        save_hidden=False, 
                        token_len=self.dst_embedding_length_a, 
                        dim=self.dst_embedding_d_a, 
                        depth=self.embedding_depth_a, 
                        heads=self.embedding_heads_a, 
                        mlp_dim=self.dst_embedding_hidden_d_a)
        )

        self.embedding_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.dst_embedding_d_v),
            Transformer(num_frames=self.orig_length_v, 
                        save_hidden=False, 
                        token_len=self.dst_embedding_length_v, 
                        dim=self.dst_embedding_d_v, 
                        depth=self.embedding_depth_v, 
                        heads=self.embedding_heads_v, 
                        mlp_dim=self.dst_embedding_hidden_d_v)
        )

        # AHL
        self.l_encoder = Transformer(
            num_frames=self.dst_embedding_length_l, 
            save_hidden=True, 
            token_len=None, 
            dim=self.dst_embedding_d_l, 
            depth=self.AHL_depth-1, 
            heads=self.l_encoder_heads, 
            mlp_dim= self.l_encoder_hidden_d)
            
        self.h_hyper_layer = HhyperLearningEncoder(
            dim=self.h_hyper_layer_d, 
            depth=self.AHL_depth, 
            heads=self.h_hyper_layer_heads, 
            dim_head=int(self.h_hyper_layer_d/self.h_hyper_layer_heads))
        
        # Fusion
        self.fusion_layer = CrossTransformer(
            source_num_frames=self.dst_embedding_length_l, 
            tgt_num_frames=self.dst_embedding_length_l, # length of l, a and v is same
            dim=self.fusion_d, 
            depth=self.fusion_layer_depth, 
            heads=self.fusion_heads, 
            mlp_dim=self.fusion_hidden_d)

        # Regression
        self.regression_head = nn.Linear(128, 1)


    def forward(self, text, audio, video):
        b = video.size(0)
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)

        x_text = self.bertmodel(text)
        h_l = self.embedding_l(x_text)[:, :8]
        h_a = self.embedding_a(audio)[:, :8]
        h_v = self.embedding_v(video)[:, :8]

        h_l_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_l_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_l_list[-1])[:, 0]

        output = self.regression_head(feat)

        return output


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_a, h_v, h_hyper):
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(h_t)
        k_ta = self.to_k_ta(h_a)
        k_tv = self.to_k_tv(h_v)
        v_ta = self.to_v_ta(h_a)
        v_tv = self.to_v_tv(h_v)

        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

        dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
        attn_ta = self.attend(dots_ta)
        out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv)
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_ta + out_tv)
        h_hyper += h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t
