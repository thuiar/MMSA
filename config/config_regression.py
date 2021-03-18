import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # single-task
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'graph_mfn': self.__Graph_MFN,
            # 'bert_mag': self.__BERT_MAG,
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
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
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
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
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
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
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
                }
            }
        }
        return tmp

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
            # dataset
            'datasetParas':{
                'mosi':{
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.2,
                    'relu_dropout': 0.2,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (40, 10), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 4,
                    'learning_rate': 1e-3,
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 4, 
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 1,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.1, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8, 
                    # when to decay learning rate (default: 20)
                    'patience': 20, 
                    'weight_decay': 0.0,
                },
                'mosei':{
                    'attn_dropout_a': 0.0,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.0,
                    'res_dropout': 0.0,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (30, 6), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 4,
                    'learning_rate': 5e-4,
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 4, 
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 5, 
                    'conv1d_kernel_size_a': 1,
                    'conv1d_kernel_size_v': 3,
                    # dropout
                    'text_dropout': 0.3, # textual Embedding Dropout
                    'attn_dropout': 0.4, # crossmodal attention block dropout
                    'output_dropout': 0.5,
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.6, 
                    # when to decay learning rate (default: 20)
                    'patience': 20, 
                    'weight_decay': 0.001,
 
                },
                'sims':{
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (50, 10), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate': 2e-3,
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 2, 
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 5, 
                    'conv1d_kernel_size_a': 1,
                    'conv1d_kernel_size_v': 1,
                    # dropout
                    'text_dropout': 0.3, # textual Embedding Dropout
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.6, 
                    # when to decay learning rate (default: 20)
                    'patience': 10, 
                    'weight_decay': 0.001,
                }
            },
        }
        return tmp
    
    # def __BERT_MAG(self):
    #     tmp = {
    #         'commonParas':{
    #             'need_align': False,
    #             'use_finetune': True,
    #             'use_bert': True,
    #             'early_stop': 12,
    #             'multi_label': False,
    #             'need_normalize': False,
    #             # Tuning
    #             'weight_decay': 0.0,
    #         },
    #         # dataset
    #         'datasetParas':{
    #             'mosi':{
    #                 # 'num_layers': 2,
    #                 # 'dropout': 0.1,
    #                 # ref Original Paper
    #                 # the batch_size of each epoch is updata_epochs * batch_size
    #                 'update_epochs': 1,
    #                 'batch_size': 32,
    #                 'learning_rate': 5e-5,
    #                 'loss_function':'ll1',
    #                 'd_acoustic_in':5,
    #                 'd_visual_in':20,
    #                 # 'h_audio_lstm':16,
    #                 # 'h_video_lstm':16,
    #                 'h_merge_sent':768,
    #                 # 'fc1_out':32,
    #                 # 'fc1_dropout':0.1,
    #                 'hidden_dropout_prob':0.2,
    #                 'beta_shift':1,
    #                 # add MAG after "AV_index" layer
    #                 # -1 means adding after all layers
    #                 # -2 means not use MAG
    #                 'AV_index':1,
    #                 'output_mode':'regression',
    #                 'num_labels':1, # is valid when output_mode == "classification"
    #             },
    #             'mosei':{
    #                 # 'num_layers': 2,
    #                 # 'dropout': 0.1,
    #                 # ref Original Paper
    #                 # the batch_size of each epoch is updata_epochs * batch_size
    #                 'update_epochs': 1,
    #                 'batch_size': 32,
    #                 'learning_rate': 5e-5,
    #                 'loss_function':'ll1',
    #                 'd_acoustic_in':74,
    #                 'd_visual_in':35,
    #                 # 'h_audio_lstm':16,
    #                 # 'h_video_lstm':16,
    #                 'h_merge_sent':768,
    #                 # 'fc1_out':32,
    #                 # 'fc1_dropout':0.1,
    #                 'hidden_dropout_prob':0.2,
    #                 'beta_shift':1,
    #                 # add MAG after "AV_index" layer
    #                 # -1 means adding after all layers
    #                 # -2 means not use MAG
    #                 'AV_index':1,
    #                 'output_mode':'regression',
    #                 'num_labels':1, # is valid when output_mode == "classification"
    #             },
    #             'sims':{
    #                 # 'num_layers': 2,
    #                 # 'dropout': 0.1,
    #                 # ref Original Paper
    #                 # the batch_size of each epoch is updata_epochs * batch_size
    #                 'update_epochs': 1,
    #                 'batch_size': 32,
    #                 'learning_rate': 5e-5,
    #                 'loss_function':'ll1',
    #                 'd_acoustic_in':33,
    #                 'd_visual_in':709,
    #                 # 'h_audio_lstm':16,
    #                 # 'h_video_lstm':16,
    #                 'h_merge_sent':768,
    #                 # 'fc1_out':32,
    #                 # 'fc1_dropout':0.1,
    #                 'hidden_dropout_prob':0.2,
    #                 'beta_shift':1,
    #                 # add MAG after "AV_index" layer
    #                 # -1 means adding after all layers
    #                 # -2 means not use MAG
    #                 'AV_index':7,
    #                 'output_mode':'regression',
    #                 'num_labels':2, # is valid when output_mode == "classification"
    #             },
    #         },
    #     }
    #     return tmp
    
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
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.8,
                    'diff_weight': 0.3,
                    'sim_weight': 0.8,
                    'sp_weight': 0.0,
                    'recon_weight': 0.8,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 1.0,
                    'weight_decay': 0.002,
                },
                'mosei':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.5,
                    'diff_weight': 0.1,
                    'sim_weight': 1.0,
                    'sp_weight': 1.0,
                    'recon_weight': 0.8,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8,
                    'weight_decay': 0.0,
                },
                'sims':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.5,
                    'diff_weight': 0.5,
                    'sim_weight': 0.5,
                    'sp_weight': 0.0,
                    'recon_weight': 0.8,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8,
                    'weight_decay': 0.0,
                }
            },
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
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 128,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                },
                'mosei':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 64,
                    'post_fusion_dim': 32,
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                },
                'sims':{
                    'hidden_dims': (128, 32, 128),
                    'text_out': 256,
                    'post_fusion_dim': 32,
                    'dropouts': (0.4, 0.4, 0.4, 0.4),
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                },
            },
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
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (256, 32, 256),
                    'dropouts': (0.3, 0.3, 0.3, 0.3),
                    'rank': 4,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'factor_lr': 1e-3, # factor_learning_rate
                    'weight_decay': 1e-4,
                },
                'mosei':{
                    'hidden_dims': (128, 16, 128),
                    'dropouts': (0.4, 0.4, 0.4, 0.4),
                    'rank': 5,
                    'batch_size': 64,
                    'learning_rate': 5e-4,
                    'factor_lr': 1e-4, # factor_learning_rate
                    'weight_decay': 5e-3,
                },
                'sims':{
                    'hidden_dims': (256, 32, 256),
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'rank': 3,
                    'batch_size': 32,
                    'learning_rate': 2e-3,
                    'factor_lr': 1e-3, # factor_learning_rate
                    'weight_decay': 5e-3,
                },
            },
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
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 32, 128),
                    'memsize': 400,
                    'windowsize': 2,
                    'NN1Config': {"drop": 0.2, "shapes": 128},
                    'NN2Config': {"drop": 0.7, "shapes": 64},
                    'gamma1Config': {"drop": 0.0, "shapes": 64},
                    'gamma2Config': {"drop": 0.0, "shapes": 256},
                    'outConfig': {"drop": 0.2, "shapes": 256},
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                },
                'mosei':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 256,
                    'windowsize': 2,
                    'NN1Config': {"drop": 0.7, "shapes": 32},
                    'NN2Config': {"drop": 0.2, "shapes": 256},
                    'gamma1Config': {"drop": 0.2, "shapes": 256},
                    'gamma2Config': {"drop": 0.0, "shapes": 256},
                    'outConfig': {"drop": 0.0, "shapes": 128},
                    'batch_size': 32,
                    'learning_rate': 2e-3,
                },
                'sims':{
                    'hidden_dims': (64, 16, 64),
                    'memsize': 64,
                    'windowsize': 2,
                    'NN1Config': {"drop": 0.2, "shapes": 128},
                    'NN2Config': {"drop": 0.0, "shapes": 128},
                    'gamma1Config': {"drop": 0.2, "shapes": 64},
                    'gamma2Config': {"drop": 0.7, "shapes": 64},
                    'outConfig': {"drop": 0.7, "shapes": 64},
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                },
            },
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
            'datasetParas':{
                'mosi':{
                    'hidden_dims': 256,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4,
                },
                'mosei':{
                    'hidden_dims': 256,
                    'num_layers': 3,
                    'dropout': 0.4,
                    'batch_size': 128,
                    'learning_rate': 5e-3,
                    'weight_decay': 1e-4,
                },
                'sims':{
                    'hidden_dims': 32,
                    'num_layers': 3,
                    'dropout': 0.5,
                    'batch_size': 128,
                    'learning_rate': 5e-4,
                    'weight_decay': 1e-3,
                },
            },
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
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 16, 64),
                    'text_out': 256,
                    'post_fusion_dim': 16,
                    'dropouts': (0.4, 0.4, 0.4, 0.4),
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'weight_decay': 0.01,
                },
                'mosei':{
                    'hidden_dims': (64, 32, 64),
                    'text_out': 256,
                    'post_fusion_dim': 16,
                    'dropouts': (0.4, 0.4, 0.4, 0.4),
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'weight_decay': 0.01,
                },
                'sims':{
                    'hidden_dims': (128, 32, 128),
                    'text_out': 256,
                    'post_fusion_dim': 32,
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'batch_size': 128,
                    'learning_rate': 5e-3,
                    'weight_decay': 5e-3,
                },
            },
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
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 32, 64),
                    'memsize': 128,
                    'inner_node_dim': 32,
                    'NNConfig': {"drop": 0.0, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 32},
                    'gamma2Config': {"drop": 0.7, "shapes": 64},
                    'outConfig': {"drop": 0.7, "shapes": 32},
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0,
                },
                'mosei':{
                    'hidden_dims': (64, 32, 64),
                    'memsize': 400,
                    'inner_node_dim': 20,
                    'NNConfig': {"drop": 0.7, "shapes": 64},
                    'gamma1Config': {"drop": 0.2, "shapes": 32},
                    'gamma2Config': {"drop": 0.5, "shapes": 64},
                    'outConfig': {"drop": 0.5, "shapes": 64},
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0,
                },
                'sims':{
                    'hidden_dims': (256, 32, 256),
                    'memsize': 64,
                    'inner_node_dim': 128,
                    'NNConfig': {"drop": 0.5, "shapes": 256},
                    'gamma1Config': {"drop": 0.2, "shapes": 32},
                    'gamma2Config': {"drop": 0.7, "shapes": 32},
                    'outConfig': {"drop": 0.2, "shapes": 256},
                    'batch_size': 64,
                    'learning_rate': 0.0005,
                    'weight_decay': 0.001,
                },
            },
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
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (256,32,256),
                    'text_out': 256,
                    'post_fusion_dim': 64,
                    'post_text_dim': 64,
                    'post_audio_dim': 4,
                    'post_video_dim': 8,
                    'dropouts': (0.2,0.2,0.2),
                    'post_dropouts': (0.2,0.2,0.2,0.2),
                    'batch_size': 64,
                    'learning_rate': 5e-3,
                    'M': 0.6,
                    'T': 0.4,
                    'A': 0.0,
                    'V': 1.0,
                    'text_weight_decay': 1e-3,
                    'audio_weight_decay': 1e-4,
                    'video_weight_decay': 1e-5,
                    'weight_decay': 0.005
                }
            },
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
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (256, 32, 256),
                    'post_text_dim': 8,
                    'post_audio_dim': 4,
                    'post_video_dim': 32,
                    # dropout
                    'post_dropouts': (0.4,0.4,0.4,0.4),
                    'dropouts': (0.3, 0.3, 0.3),
                    'rank': 3,
                    'batch_size': 128,
                    'learning_rate': 0.001,
                    'factor_lr': 0.001, # factor_learning_rate
                    'M': 0.2,
                    'T': 1.0,
                    'A': 0.4,
                    'V': 0.8,
                    'text_weight_decay': 1e-4,
                    'audio_weight_decay': 0.0,
                    'video_weight_decay': 1e-4, 
                    'weight_decay': 0.005,
                },
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
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (64, 32, 64),
                    'text_out': 32,
                    'post_fusion_dim': 32,
                    'post_text_dim': 16,
                    'post_audio_dim': 4,
                    'post_video_dim': 32,
                    'dropouts': (0.3, 0.3, 0.3),
                    'post_dropouts': (0.5,0.5,0.5,0.5),
                    'batch_size': 32,
                    'learning_rate': 0.002,
                    'M': 0.6,
                    'T': 0.2,
                    'A': 0.8,
                    'V': 0.8,
                    'text_weight_decay': 0.0001,
                    'audio_weight_decay': 0.001,
                    'video_weight_decay': 0.0001,
                    'weight_decay': 0.0,
                }
            },
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
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 32,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.01,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.01,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args