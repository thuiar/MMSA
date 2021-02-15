import os
import random
import argparse

from utils.functions import Storage


class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # single-task
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'graph_mfn': self.__Graph_MFN,
            'mag': self.__MAG,
            'mult': self.__MULT,
            'misa': self.__MISA,
            # multi-task
            'mtfn': self.__MTFN,
            'mlmf': self.__MLMF,
            'mlf_dnn': self.__MLF_DNN,
            'self_mm': self.__SELF_MM
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/home/sharing/disk3/dataset/multimodal-sentiment-dataset/StandardDatasets'
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
                'attn_dropout_a': random.choice([0.0, 0.1, 0.2]),
                'attn_dropout_v': random.choice([0.0, 0.1, 0.2]),
                'relu_dropout': random.choice([0.0, 0.1, 0.2]),
                'embed_dropout': random.choice([0.0, 0.1, 0.2]),
                'res_dropout': random.choice([0.0, 0.1, 0.2]),
                #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                'dst_feature_dim_nheads': random.choice([(30,6),(30,10),(32,8),(36,6),(40,5),(40,8),(40,10),(50,10)]),
                # the batch_size of each epoch is updata_epochs * batch_size
                'batch_size': random.choice([4,8,16]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                # number of layers(Blocks) in the Crossmodal Networks
                'nlevels': random.choice([2,4,6]), 
                # temporal convolution kernel size
                'conv1d_kernel_size_l': random.choice([1,3,5]), 
                'conv1d_kernel_size_a': random.choice([1,3,5]),
                'conv1d_kernel_size_v': random.choice([1,3,5]),
                # dropout
                'text_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]), # textual Embedding Dropout
                'attn_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]), # crossmodal attention block dropout
                'output_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]),
                'grad_clip': random.choice([0.6,0.8,1.0]), # gradient clip value (default: 0.8)
                'patience': random.choice([5, 10, 20]),
                'weight_decay': random.choice([0.0,0.001,0.005]),
            }
        }
        return tmp
    
    def __MAG(self):
        tmp = {
            'commonParas':{
                'need_align': False,
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
                'exclude_zero': random.choice([True, False]),
                'AV_index': random.choice([-2, 1]),
                'batch_size': random.choice([32, 48, 64]),
                'learning_rate': random.choice([2e-5, 5e-5, 1e-4, 1e-5]),
                'hidden_dropout_prob': random.choice([0.2, 0.1, 0.5]),
                'beta_shift': random.choice([1]),
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
                'batch_size': random.choice([16, 32, 64]),
                'learning_rate': random.choice([1e-4, 1e-3, 5e-4]),
                'hidden_size': random.choice([64, 128, 256]),
                'dropout': random.choice([0.5, 0.2, 0.0]),
                'reverse_grad_weight': random.choice([0.5, 0.8, 1.0]),
                'diff_weight': random.choice([0.1, 0.3, 0.5]),
                'sim_weight': random.choice([0.5, 0.8, 1.0]),
                'sp_weight': random.choice([0.0, 1.0]),
                'recon_weight': random.choice([0.5, 0.8, 1.0]),
                # when grad_clip == -1.0, means not use that
                'grad_clip': random.choice([-1.0, 0.8, 1.0]),
                'weight_decay': random.choice([0.0, 5e-5, 2e-3])
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)]),
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'dropouts': random.choice([(0.3, 0.3, 0.3, 0.5),(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4)]),
                'rank': random.choice([3,4,5,6]),
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'factor_lr': random.choice([1e-4,5e-4,1e-3]), # factor_learning_rate
                'weight_decay': random.choice([0.0, 1e-4,1e-3,5e-3]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'memsize': random.choice([64,128,256,300,400]),
                'windowsize': 2,
                'NN1Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'NN2Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'gamma1Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'gamma2Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'outConfig': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
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
                'hidden_dims': random.choice([16,32,64,128,256]),
                'num_layers': random.choice([2,3,4]),
                'dropout': random.choice([0.2,0.3,0.4,0.5]),
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'weight_decay': random.choice([0.0, 1e-4,1e-3,5e-3])
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)]),
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'weight_decay': random.choice([0.0,1e-3,5e-3,1e-2]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'memsize': random.choice([64,128,256,300,400]),
                'inner_node_dim': random.choice([20, 32, 64, 128]),
                'NNConfig': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'gamma1Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'gamma2Config': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'outConfig': {"drop": random.choice([0.0,0.2,0.5,0.7]), "shapes": random.choice([32,64,128,256])},
                'batch_size': random.choice([32,64]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'weight_decay': random.choice([0.0,1e-3,5e-3,1e-2]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([4,5]),
                'post_video_dim': random.choice([8,16,32,64]),
                'dropouts': random.choice([(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4),(0.3, 0.3, 0.3)]),
                'post_dropouts': random.choice([(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)]),
                # # ref Original Paper
                'batch_size': random.choice([32,64]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                # ref Original Paper
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'text_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'audio_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'video_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'weight_decay': random.choice([0.0,1e-3,5e-3,1e-2]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([4,5]),
                'post_video_dim': random.choice([8,16,32,64]),
                'post_dropouts': random.choice([(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)]),
                'dropouts': random.choice([(0.5, 0.5, 0.5),(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4)]),
                'rank': random.choice([3,4,5,6]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'factor_lr': random.choice([1e-4,5e-4,1e-3]), # factor_learning_rate
                # dropout
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'text_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'audio_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'video_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'weight_decay': random.choice([0.0, 1e-4,1e-3,5e-3]),
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
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([4,5]),
                'post_video_dim': random.choice([8,16,32,64]),
                'dropouts': random.choice([(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4),(0.3, 0.3, 0.3)]),
                'post_dropouts': random.choice([(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)]),
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'text_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'audio_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'video_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'weight_decay': random.choice([0.0, 1e-4,1e-3,5e-3]),
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
                'batch_size': random.choice([16, 32]),
                'learning_rate_bert': random.choice([5e-5]),
                'learning_rate_audio': random.choice([1e-4, 1e-3, 5e-3]),
                'learning_rate_video': random.choice([1e-4, 1e-3, 5e-3]),
                'learning_rate_other': random.choice([1e-4, 1e-3]),
                'weight_decay_bert': random.choice([0.001, 0.01]),
                'weight_decay_audio': random.choice([0.0, 0.001, 0.01]),
                'weight_decay_video': random.choice([0.0, 0.001, 0.01]),
                'weight_decay_other': random.choice([0.001, 0.01]),
                # feature subNets
                'a_lstm_hidden_size': random.choice([16, 32]),
                'v_lstm_hidden_size': random.choice([32, 64]),
                'a_lstm_layers': 1,
                'v_lstm_layers': 1,
                'text_out': 768,
                'audio_out': random.choice([16]),
                'video_out': random.choice([32]), 
                'a_lstm_dropout': random.choice([0.0]),
                'v_lstm_dropout': random.choice([0.0]),
                't_bert_dropout':random.choice([0.1]),
                # post feature
                'post_fusion_dim': random.choice([64, 128]),
                'post_text_dim':random.choice([32, 64]),
                'post_audio_dim': random.choice([16, 32]),
                'post_video_dim': random.choice([16, 32]),
                'post_fusion_dropout': random.choice([0.1, 0.0]),
                'post_text_dropout': random.choice([0.1, 0.0]),
                'post_audio_dropout': random.choice([0.1, 0.0]),
                'post_video_dropout': random.choice([0.1, 0.0]),
            }
        }
        return tmp

    def get_config(self):
        return self.args

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug_mode', type=bool, default=False,
                            help='adjust parameters ?')
        parser.add_argument('--modelName', type=str, default='ef_lstm',
                            help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/graph_mfn/mtfn/mlmf/mlf_dnn')
        parser.add_argument('--datasetName', type=str, default='sims',
                            help='support mosi/sims')
        parser.add_argument('--tasks', type=str, default='M',
                            help='M/T/A/V/MTAV/...')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num workers of loading data')
        parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                            help='path to save model.')
        parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                            help='path to save results.')
        parser.add_argument('--data_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset',
                            help='path to data directory')
        parser.add_argument('--gpu_ids', type=list, default=[2],
                            help='indicates the gpus will be used.')
        return parser.parse_args()
        
    args = parse_args()
    config = ConfigTune(args)
    args = config.get_config()
    print(args)