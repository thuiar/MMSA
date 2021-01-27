"""
paper: Integrating MultimodalInformationinLargePretrainedTransformers
From: https://github.com/WasifurRahman/BERT_multimodal_transformer/tree/optuna
"""
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertLayerNorm, BertEmbeddings, BertPooler, BertLayer, BertForSequenceClassification, BertConfig, MultimodalBertForSequenceClassification
from pytorch_transformers.modeling_utils import PretrainedConfig
from pytorch_transformers.amir_tokenization import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

__all__ = ['BERT_MAG']

class MAG(nn.Module):
    def __init__(self, config, newly_added_config):
        super(MAG, self).__init__()
        self.W_hv = nn.Linear(newly_added_config["d_visual_in"]+newly_added_config["h_merge_sent"],newly_added_config["h_merge_sent"])
        self.W_ha = nn.Linear(newly_added_config["d_acoustic_in"]+newly_added_config["h_merge_sent"],newly_added_config["h_merge_sent"])
        self.W_v = nn.Linear(newly_added_config["d_visual_in"],newly_added_config["h_merge_sent"])
        self.W_a = nn.Linear(newly_added_config["d_acoustic_in"],newly_added_config["h_merge_sent"])
        self.beta = newly_added_config["beta_shift"]
        self.newly_added_config = newly_added_config
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-6)
        self.final_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, text_embedding,acoustic,visual):
        eps = 1e-6
        weight_v= F.relu(self.W_hv(torch.cat((visual,text_embedding),dim=-1)))
        weight_a= F.relu(self.W_ha(torch.cat((acoustic,text_embedding),dim=-1)))
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        em_norm = text_embedding.norm(2,dim=-1)
        hm_norm = h_m.norm(2,dim=-1)
        hm_norm_ones=torch.ones(hm_norm.shape,requires_grad=True).to(self.newly_added_config["device"])
        hm_norm=torch.where(hm_norm==0,hm_norm_ones,hm_norm)
        thresh_hold = (em_norm/(hm_norm+eps))*self.beta
        ones = torch.ones(thresh_hold.shape,requires_grad=True).to(self.newly_added_config["device"])
        alpha = torch.min(thresh_hold,ones)
        alpha=alpha.unsqueeze(dim=-1)
        acoustic_vis_embedding = alpha*h_m
        embedding_output = self.final_dropout(self.LayerNorm(acoustic_vis_embedding + text_embedding))
        return embedding_output

class BERT_MAG_model(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_MAG_model, self).__init__(config)
        self.newly_added_config = args
        if args.output_mode == 'regression':
            self.num_labels = 1
        #BertEncoder
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        self.MAG = MAG(self.config,args)
        self.MAG_all = nn.ModuleList([MAG(self.config,args) for _ in range(self.config.num_hidden_layers)])
        
        # MultimodalBertModel
        self.embeddings = BertEmbeddings(self.config)
        self.pooler = BertPooler(self.config)

        # MultimodalBertForSequenceClassification
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(args["hidden_dropout_prob"])
        self.apply(self.init_weights)

    def Encoder(self, hidden_states, visual, acoustic, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        AV_index = self.newly_added_config["AV_index"]
        for i, layer_module in enumerate(self.layer):
            if AV_index >= 0 and i == AV_index:
                hidden_states = self.MAG(hidden_states,acoustic,visual)
            elif AV_index == -1:
                hidden_states = self.MAG_all[i](hidden_states,acoustic,visual)
            elif AV_index == -2:
                hidden_states = hidden_states

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs 


    def forward(self, text, audio, visual, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None):
        input_mask = text[:, 1, :].squeeze().long()
        segment_ids = text[:, 2, :].squeeze().long()
        input_ids = text[:, 0, :].squeeze().long()

        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, segment_ids)
        encoder_outputs = self.Encoder(embedding_output,visual,audio,extended_attention_mask,head_mask=head_mask)
        sequence_output = encoder_outputs[0] # batch * length * hidden size
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # MultimodalBertForSequenceClassification
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs

class BERT_MAG(nn.Module):
    def __init__(self, args):
        super(BERT_MAG, self).__init__()
        if args.language == 'cn':
            pretrained_dir = 'pretrained_model/bert_cn'
        elif args.language == 'en':
            pretrained_dir = 'pretrained_model/bert_en'

        self.config = self.get_config(pretrained_dir,newly_added_config = args, cache_dir='', num_labels=1)
        self.model = BERT_MAG_model(self.config, args)
        archive_file = os.path.join(os.path.join(pretrained_dir, 'pytorch_model.bin'))
        state_dict = torch.load(archive_file, map_location='cpu')

        # update params name
        key_map = collections.OrderedDict({'gamma': 'weight', 'beta': 'bias', 'bert.encoder.': '', 'bert.': ''})
        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = key
            for old, new in key_map.items():
                if old in new_key:
                    new_key = new_key.replace(old, new)
            if new_key != key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
           
        # load pre-trained params
        self.model.load_state_dict(state_dict, strict=False)

    def get_config(self, pretrained_model_name_or_path, newly_added_config=None, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        output_loading_info = kwargs.pop('output_loading_info', False)

        config, _ = MultimodalBertForSequenceClassification.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                **kwargs
            )
        return config

    def forward(self, text, audio, visual):
        outputs = self.model(text, audio, visual)
        return outputs