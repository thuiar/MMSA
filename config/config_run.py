import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

__all__ = ['Config']

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

class Config():
    def __init__(self, args):
        # parameters for data
        self.data_dir = args.data_dir
        # global parameters for running
        try:
            self.global_running = vars(args)
        except TypeError:
            self.global_running = args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            'mult': self.__MULT,
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'mtfn': self.__MTFN,
            'mlmf': self.__MLMF,
            'mlf_dnn': self.__MLF_DNN,
        }
        # hyper parameters for datasets
        self.HYPER_DATASET_MAP = self.__datasetCommonParams()
    
    def __datasetCommonParams(self):
        tmp = {
            'mosi':{
                'datapath': os.path.join(self.data_dir, 'MOSI/Processed/CMU-SDK/seq_length_50/mosi_data_noalign.pkl'),
                # (batch_size, input_lens, feature_dim)
                'input_lens': (50, 50, 50),
                'feature_dims': (300, 5, 20), # (text, audio, video)
            },
            'sims':{
                'datapath': os.path.join(self.data_dir, 'CH-SIMS/Processed/features/data.npz'),
                'label_dir': os.path.join(self.data_dir, 'CH-SIMS/metadata'),
                'nsamples': 2281,
                # (batch_size, input_lens, feature_dim)
                'input_lens': (39, 400, 55), # (text, audio, video)
                'feature_dims': (768, 33, 709), # (text, audio, video)
            }
        }
        return tmp

    # baselines
    def __MULT(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                # Task
                'vonly': True, # use the crossmodal fusion into v
                'aonly': True, # use the crossmodal fusion into v
                'lonly': True, # use the crossmodal fusion into v
                'aligned': False, # consider aligned experiment or not
                # Architecture
                'attn_mask': True, # use attention mask for Transformer
                'attn_dropout_a': 0.0,
                'attn_dropout_v': 0.0,
                'relu_dropout': 0.1,
                'embed_dropout': 0.25,
                'res_dropout': 0.1,
                # Tuning
                'early_stop': 20,
                'patience': 8, # when to decay learning rate (default: 20)
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (30,10), 
                    'batch_size': 16,
                    'learning_rate': 1e-3,
                    'num_epochs': 100,
                    'nlevels': 4, # number of layers(Blocks) in the Crossmodal Networks
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 3,
                    'conv1d_kernel_size_v': 3,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    'grad_clip': 0.8, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'criterion': 'L1Loss',
                    'num_classes': 1,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (50,10), 
                    # ref Original Paper
                    'batch_size': 24,
                    'learning_rate': 5e-3,
                    'nlevels': 2, # number of layers(Blocks) in the Crossmodal Networks
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 3, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 1,
                    # dropout
                    'text_dropout': 0.3, # textual Embedding Dropout
                    'attn_dropout': 0.4, # crossmodal attention block dropout
                    'output_dropout': 0.2,
                    'grad_clip': 0.6, # gradient clip value (default: 0.8)
                }
            },
        }
        return tmp
    
    def __TFN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (256, 32, 256),
                    'text_out': 64,
                    'post_fusion_dim': 16,
                    'dropouts': (0.2, 0.2, 0.2, 0.3),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128,16,128),
                    'text_out': 128,
                    'post_fusion_dim': 64,
                    'dropouts': (0.2,0.2,0.2,0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 2e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                }
            },
        }
        return tmp
    
    def __LMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 16, 64),
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 5,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 2e-3,
                    'factor_lr': 1e-4, # factor_learning_rate
                    'weight_decay': 1e-4,
                    'grad_clip': 0.0, # gradient clip value
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 4,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'factor_lr': 5e-4, # factor_learning_rate
                    'weight_decay': 1e-3,
                    'grad_clip': 0.0, # gradient clip value
                },
            },
        }
        return tmp
    
    def __MFN(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 4, 16),
                    'memsize': 64,
                    'windowsize': 2,
                    'output_dim': 1,
                    'NN1Config': {"drop": 0.7, "shapes": 256},
                    'NN2Config': {"drop": 0.7, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 128},
                    'gamma2Config': {"drop": 0.2, "shapes": 32},
                    'outConfig': {"drop": 0.5, "shapes": 128},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 16,
                    'learning_rate': 1e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 64,
                    'windowsize': 2,
                    'output_dim': 1,
                    'NN1Config': {"drop": 0.2, "shapes": 32},
                    'NN2Config': {"drop": 0.7, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 256},
                    'gamma2Config': {"drop": 0.7, "shapes": 32},
                    'outConfig': {"drop": 0.2, "shapes": 32},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
        }
        return tmp
    
    def __EF_LSTM(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': 64,
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'num_layers': 2,
                    'dropout': 0.3,
                    # ref Original Paper
                    'batch_size': 16,
                    'learning_rate': 1e-3,
                },
                'sims':{
                    'hidden_dims': 16,
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'num_layers': 2,
                    'dropout': 0.4,
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                },
            },
        }
        return tmp
    
    def __LF_DNN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
        }
        return tmp
   
    def __MTFN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'multi_label': True,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (256,32,256),
                    'text_out': 128,
                    'post_fusion_dim': 64,
                    'post_text_dim': 16,
                    'post_audio_dim': 4,
                    'post_video_dim': 8,
                    'dropouts': (0.3,0.3,0.3),
                    'post_dropouts': (0.4,0.4,0.4,0.4),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                    'M': 0.2,
                    'T': 0.4,
                    'A': 1.0,
                    'V': 0.6,
                    'text_weight_decay': 1e-5,
                    'audio_weight_decay': 1e-3,
                    'video_weight_decay': 1e-4,
                }
            },
        }
        return tmp

    def __MLMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (64, 16, 64),
                    'post_text_dim': 64,
                    'post_audio_dim': 5,
                    'post_video_dim': 16,
                    # dropout
                    'post_dropouts': (0.3,0.3,0.3),
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 4,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'factor_lr': 0.0005, # factor_learning_rate
                    'weight_decay': 0.001,
                    'grad_clip': 0.0, # gradient clip value
                    'M': 0.4,
                    'T': 0.6,
                    'A': 1.0,
                    'V': 0.2,
                    'text_weight_decay': 1e-4,
                    'audio_weight_decay': 1e-5,
                    'video_weight_decay': 1e-4, 
                },
            },
        }
        return tmp
    
    def __MLF_DNN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'multi_label': True,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
                'KeyEval': 'Mult_acc_2',
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 5,
                    'post_video_dim': 16,
                    'dropouts': (0.2, 0.2, 0.2),
                    'post_dropouts': (0.5,0.5,0.5,0.5),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.0005,
                    'grad_clip': 0.0,
                    'M': 0.8,
                    'T': 0.6,
                    'A': 0.4,
                    'V': 0.2,
                    'text_weight_decay': 0.0,
                    'audio_weight_decay': 0.0,
                    'video_weight_decay': 0.0,
                },
            },
        }
        return tmp

    def get_config(self):
        # normalize
        model_name = str.lower(self.global_running['modelName'])
        dataset_name = str.lower(self.global_running['datasetName'])
        # integrate all parameters
        res =  Storage(dict(self.global_running,
                            **self.HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            **self.HYPER_MODEL_MAP[model_name]()['commonParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res