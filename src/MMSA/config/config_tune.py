import os
import random
import argparse
from easydict import EasyDict as edict
import json
# from utils.functions import Storage


class ConfigTune():
    def __init__(self):
        # global parameters for running
        # self.globalArgs = args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            # single-task
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'graph_mfn': self.__Graph_MFN,
            'mag': self.__BERT_MAG,
            'mult': self.__MULT,
            'misa': self.__MISA,
            'mfm': self.__MFM,
            # multi-task
            'mtfn': self.__MTFN,
            'mlmf': self.__MLMF,
            'mlf_dnn': self.__MLF_DNN,
            'self_mm': self.__SELF_MM
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.model_name)
        dataset_name = str.lower(args.dataset_name)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = edict(
            vars(args),
            **dataArgs,
            **commonArgs,
            **HYPER_MODEL_MAP[model_name]()['debugParas'],
        )
    
    def __datasetCommonParams(self):
        root_dataset_dir = ''
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'H': 1.0
                }
            }
        }
        return tmp

    # baselines
    def __MULT(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'early_stop': 8,
                'use_bert': False,
                # use finetune for bert
                'use_bert_finetune': False,
                # use attention mask for Transformer
                'attn_mask': True, 
                'update_epochs': 8,
            },
            'debugParas':{
                'd_paras': ['attn_dropout_a', 'attn_dropout_v', 'relu_dropout', 'embed_dropout', 'res_dropout',\
                    'dst_feature_dim_nheads','batch_size','learning_rate','nlevels',\
                    'conv1d_kernel_size_l','conv1d_kernel_size_a','conv1d_kernel_size_v','text_dropout',\
                        'attn_dropout','output_dropout','grad_clip', 'patience', 'weight_decay'],
                'attn_dropout_a': [0.0, 0.1, 0.2],
                'attn_dropout_v': [0.0, 0.1, 0.2],
                'relu_dropout': [0.0, 0.1, 0.2],
                'embed_dropout': [0.0, 0.1, 0.2],
                'res_dropout': [0.0, 0.1, 0.2],
                #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                'dst_feature_dim_nheads': [(30,6),(30,10),(32,8),(36,6),(40,5),(40,8),(40,10),(50,10)],
                # the batch_size of each epoch is updata_epochs * batch_size
                'batch_size': [4,8,16],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                # number of layers(Blocks) in the Crossmodal Networks
                'nlevels': [2,4,6], 
                # temporal convolution kernel size
                'conv1d_kernel_size_l': [1,3,5], 
                'conv1d_kernel_size_a': [1,3,5],
                'conv1d_kernel_size_v': [1,3,5],
                # dropout
                'text_dropout': [0.1,0.2,0.3,0.4,0.5], # textual Embedding Dropout
                'attn_dropout': [0.1,0.2,0.3,0.4,0.5], # crossmodal attention block dropout
                'output_dropout': [0.1,0.2,0.3,0.4,0.5],
                'grad_clip': [0.6,0.8,1.0], # gradient clip value (default: 0.8)
                'patience': [5, 10, 20],
                'weight_decay': [0.0,0.001,0.005],
            }
        }
        return tmp
    
    def __BERT_MAG(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 8,
                'multi_label': True,
                'need_normalize': False,
                # Tuning
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'raw_mosi':{
                    'loss_function':'ll1',
                    'd_acoustic_in':5,
                    'd_visual_in':20,
                    'h_merge_sent':768,
                    # add MAG after "AV_index" layer
                    # -1 means adding after all layers
                    # -2 means not use MAG
                    # 'AV_index':1,
                    'output_mode':'regression',
                    'num_labels':2, # is valid when output_mode == "classification"
                },
            },
            'debugParas':{
                'd_paras': ['exclude_zero', 'AV_index', 'batch_size', 'learning_rate', 'hidden_dropout_prob', 'beta_shift'],
                'exclude_zero': [True, False],
                'AV_index': [-2, 1],
                'batch_size': [32, 48, 64],
                'learning_rate': [2e-5, 5e-5, 1e-4, 1e-5],
                'hidden_dropout_prob': [0.2, 0.1, 0.5],
                'beta_shift': [1],
            }
        }
        return tmp

    def __MISA(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 8,
                'update_epochs': 2,
                'rnncell': 'lstm',
                'use_cmd_sim': True,
            },
            'debugParas':{
                'd_paras': ['batch_size', 'learning_rate', 'hidden_size', 'dropout', 'reverse_grad_weight', \
                            'diff_weight', 'sim_weight', 'sp_weight', 'recon_weight', 'grad_clip', 'weight_decay'],
                'batch_size': [16, 32, 64],
                'learning_rate': [1e-4, 1e-3, 5e-4],
                'hidden_size': [64, 128, 256],
                'dropout': [0.5, 0.2, 0.0],
                'reverse_grad_weight': [0.5, 0.8, 1.0],
                'diff_weight': [0.1, 0.3, 0.5],
                'sim_weight': [0.5, 0.8, 1.0],
                'sp_weight': [0.0, 1.0],
                'recon_weight': [0.5, 0.8, 1.0],
                # when grad_clip == -1.0, means not use that
                'grad_clip': [-1.0, 0.8, 1.0],
                'weight_decay': [0.0, 5e-5, 2e-3]
            }
        }
        return tmp

    def __TFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','dropouts','batch_size','learning_rate'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(64,32,64)],
                'text_out': [32,64,128,256],
                'post_fusion_dim': [16,32,64,128],
                'dropouts': [(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)],
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
            }
        }
        return tmp
    
    def __LMF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims','dropouts','rank','batch_size','learning_rate','factor_lr','weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'dropouts': [(0.3, 0.3, 0.3, 0.5),(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4)],
                'rank': [3,4,5,6],
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'factor_lr': [1e-4,5e-4,1e-3], # factor_learning_rate
                'weight_decay': [0.0, 1e-4,1e-3,5e-3],
            }
        }
        return tmp
    
    def __MFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': True,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','memsize','windowsize','NN1Config','NN2Config','gamma1Config','gamma2Config',\
                    'outConfig','batch_size','learning_rate'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'memsize': [64,128,256,300,400],
                'windowsize': 2,
                'NN1Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'NN2Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma1Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma2Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'outConfig': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
            }
        }
        return tmp

    def __MFM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': True,
                'early_stop': 8,
            },
            
            'debugParas':{
                'd_paras': ['hidden_dims', 'zy_size', 'zl_size', 'za_size', 'zv_size', 'fy_size',\
                'fl_size', 'fa_size', 'fv_size', 'zy_to_fy_dropout', 'zl_to_fl_dropout', 'za_to_fa_dropout', 'zv_to_fv_dropout', 'fy_to_y_dropout',\
                'lda_mmd', 'lda_xl', 'lda_xa', 'lda_xv', 'memsize','windowsize','NN1Config','NN2Config','gamma1Config','gamma2Config',\
                'outConfig','batch_size','learning_rate'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'zy_size': [8,16,32,48,64,80],
                'zl_size': [32,64,88,128,156,256],
                'za_size': [8,16,32,48,64,80],
                'zv_size': [8,16,32,48,64,80],
                'fy_size': [8,16,32,48,64,80],
                'fl_size': [32,64,88,128,156,256],
                'fa_size': [8,16,32,48,64,80],
                'fv_size': [8,16,32,48,64,80],
                'zy_to_fy_dropout': [0.0,0.2,0.5,0.7],
                'zl_to_fl_dropout': [0.0,0.2,0.5,0.7],
                'za_to_fa_dropout': [0.0,0.2,0.5,0.7],
                'zv_to_fv_dropout': [0.0,0.2,0.5,0.7],
                'fy_to_y_dropout': [0.0,0.2,0.5,0.7],
                'lda_mmd': [10,50,100,200],
                'lda_xl': [0.01,0.1,0.5,1.0,2.0,5.0,10.0],
                'lda_xa': [0.01,0.1,0.5,1.0,2.0,5.0,10.0],
                'lda_xv': [0.01,0.1,0.5,1.0,2.0,5.0,10.0],
                'memsize': [64,128,256,300,400],
                'windowsize': 2,
                'NN1Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'NN2Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma1Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma2Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'outConfig': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
            }
        }
        return tmp
    
    def __EF_LSTM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': False,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','num_layers','dropout','batch_size','learning_rate', 'weight_decay'],
                'hidden_dims': [16,32,64,128,256],
                'num_layers': [2,3,4],
                'dropout': [0.2,0.3,0.4,0.5],
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'weight_decay': [0.0, 1e-4,1e-3,5e-3]
            }
        }
        return tmp
    
    def __LF_DNN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim',\
                            'dropouts','batch_size','learning_rate','weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'text_out': [32,64,128,256],
                'post_fusion_dim': [16,32,64,128],
                'dropouts': [(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)],
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'weight_decay': [0.0,1e-3,5e-3,1e-2],
            }
        }
        return tmp

    def __Graph_MFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': False,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','memsize','inner_node_dim', 'NNConfig','gamma1Config','gamma2Config',\
                    'outConfig','batch_size','learning_rate', 'weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'memsize': [64,128,256,300,400],
                'inner_node_dim': [20, 32, 64, 128],
                'NNConfig': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma1Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'gamma2Config': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'outConfig': {"drop": [0.0,0.2,0.5,0.7], "shapes": [32,64,128,256]},
                'batch_size': [32,64],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'weight_decay': [0.0,1e-3,5e-3,1e-2],
            }
        }
        return tmp
    
    def __MTFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','post_text_dim','post_audio_dim',\
                            'post_video_dim','dropouts','post_dropouts','batch_size','learning_rate',\
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay', 'weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'text_out': [32,64,128,256],
                'post_fusion_dim': [16,32,64,128],
                'post_text_dim': [8,16,32,64],
                'post_audio_dim': [4,5],
                'post_video_dim': [8,16,32,64],
                'dropouts': [(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4),(0.3, 0.3, 0.3)],
                'post_dropouts': [(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)],
                # # ref Original Paper
                'batch_size': [32,64],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                # ref Original Paper
                'M':[0.2,0.4,0.6,0.8,1],
                'T':[0,0.2,0.4,0.6,0.8,1],
                'A':[0,0.2,0.4,0.6,0.8,1],
                'V':[0,0.2,0.4,0.6,0.8,1],
                'text_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'audio_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'video_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'weight_decay': [0.0,1e-3,5e-3,1e-2],
            }
        }
        return tmp

    def __MLMF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','dropouts','rank','batch_size','learning_rate','factor_lr',\
                            'post_text_dim', 'post_audio_dim', 'post_video_dim', 'post_dropouts',\
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay','weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'post_text_dim': [8,16,32,64],
                'post_audio_dim': [4,5],
                'post_video_dim': [8,16,32,64],
                'post_dropouts': [(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)],
                'dropouts': [(0.5, 0.5, 0.5),(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4)],
                'rank': [3,4,5,6],
                # ref Original Paper
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'factor_lr': [1e-4,5e-4,1e-3], # factor_learning_rate
                # dropout
                'M':[0.2,0.4,0.6,0.8,1],
                'T':[0,0.2,0.4,0.6,0.8,1],
                'A':[0,0.2,0.4,0.6,0.8,1],
                'V':[0,0.2,0.4,0.6,0.8,1],
                'text_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'audio_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'video_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'weight_decay': [0.0, 1e-4,1e-3,5e-3],
            },
        }
        return tmp
    
    def __MLF_DNN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8,
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','post_text_dim','post_audio_dim',\
                            'post_video_dim','dropouts','post_dropouts','batch_size','learning_rate',
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay', 'weight_decay'],
                'hidden_dims': [(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)],
                'text_out': [32,64,128,256],
                'post_fusion_dim': [16,32,64,128],
                'post_text_dim': [8,16,32,64],
                'post_audio_dim': [4,5],
                'post_video_dim': [8,16,32,64],
                'dropouts': [(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4),(0.3, 0.3, 0.3)],
                'post_dropouts': [(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)],
                'batch_size': [32,64,128],
                'learning_rate': [5e-4,1e-3,2e-3,5e-3],
                'M':[0.2,0.4,0.6,0.8,1],
                'T':[0,0.2,0.4,0.6,0.8,1],
                'A':[0,0.2,0.4,0.6,0.8,1],
                'V':[0,0.2,0.4,0.6,0.8,1],
                'text_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'audio_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'video_weight_decay': [0, 1e-3, 1e-4, 1e-5],
                'weight_decay': [0.0, 1e-4,1e-3,5e-3],
            }
        }
        return tmp
    
    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4,
            },
            'debugParas':{
                'd_paras': ['batch_size', 'learning_rate_bert','learning_rate_audio', 'learning_rate_video', \
                            'learning_rate_other', 'weight_decay_bert', 'weight_decay_other', 
                            'weight_decay_audio', 'weight_decay_video',\
                            'a_lstm_hidden_size', 'v_lstm_hidden_size', 'text_out', 'audio_out', 'video_out',\
                            'a_lstm_dropout', 'v_lstm_dropout', 't_bert_dropout', 'post_fusion_dim', 'post_text_dim', 'post_audio_dim', \
                            'post_video_dim', 'post_fusion_dropout', 'post_text_dropout', 'post_audio_dropout', 'post_video_dropout', 'H'],
                'batch_size': [16, 32],
                'learning_rate_bert': [5e-5],
                'learning_rate_audio': [1e-4, 1e-3, 5e-3],
                'learning_rate_video': [1e-4, 1e-3, 5e-3],
                'learning_rate_other': [1e-4, 1e-3],
                'weight_decay_bert': [0.001, 0.01],
                'weight_decay_audio': [0.0, 0.001, 0.01],
                'weight_decay_video': [0.0, 0.001, 0.01],
                'weight_decay_other': [0.001, 0.01],
                # feature subNets
                'a_lstm_hidden_size': [16, 32],
                'v_lstm_hidden_size': [32, 64],
                'a_lstm_layers': 1,
                'v_lstm_layers': 1,
                'text_out': 768,
                'audio_out': [16],
                'video_out': [32], 
                'a_lstm_dropout': [0.0],
                'v_lstm_dropout': [0.0],
                't_bert_dropout':[0.1],
                # post feature
                'post_fusion_dim': [64, 128],
                'post_text_dim':[32, 64],
                'post_audio_dim': [16, 32],
                'post_video_dim': [16, 32],
                'post_fusion_dropout': [0.1, 0.0],
                'post_text_dropout': [0.1, 0.0],
                'post_audio_dropout': [0.1, 0.0],
                'post_video_dropout': [0.1, 0.0],
            }
        }
        return tmp

    def get_config(self):
        return self.args
    
    def get_all_config(self):
        res = {}
        res['datasetCommonParams'] = self.__datasetCommonParams()
        for key in self.HYPER_MODEL_MAP.keys():
            res[key] = self.HYPER_MODEL_MAP[key]()
        with open("./config_tune.json", "w") as f:
            json.dump(res, f, indent=2)

if __name__ == "__main__":
        
    config = ConfigTune()
    config.get_all_config()