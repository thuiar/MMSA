import os
import random
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

__all__ = ['ConfigDebug']

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

class ConfigDebug():
    def __init__(self, input_args):
        # parameters for data
        self.data_dir = '/home/sharing/disk2/multimodal-sentiment-dataset'
        # global parameters for running
        self.global_running = vars(input_args)
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
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                },
                'sims':{
                    'criterion': 'L1Loss',
                    'num_classes': 1,
                }
            },
            'debugParas':{
                'd_paras': ['dst_feature_dim_nheads','batch_size','learning_rate','nlevels',\
                    'conv1d_kernel_size_l','conv1d_kernel_size_a','conv1d_kernel_size_v','text_dropout',\
                        'attn_dropout','output_dropout','grad_clip'],
                #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                'dst_feature_dim_nheads': random.choice([(30,6),(30,10),(32,8),(36,6),(40,5),(40,8),(40,10),(50,10)]),
                # ref Original Paper
                'batch_size': random.choice([8,16,24]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'nlevels': random.choice([2,4,6]), # number of layers(Blocks) in the Crossmodal Networks
                # temporal convolution kernel size
                'conv1d_kernel_size_l': random.choice([1,3,5]), 
                'conv1d_kernel_size_a': random.choice([1,3,5]),
                'conv1d_kernel_size_v': random.choice([1,3,5]),
                # dropout
                'text_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]), # textual Embedding Dropout
                'attn_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]), # crossmodal attention block dropout
                'output_dropout': random.choice([0.1,0.2,0.3,0.4,0.5]),
                'grad_clip': random.choice([0.6,0.8,1.0]), # gradient clip value (default: 0.8)
            }
        }
        return tmp
    
    def __TFN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'num_classes': 1, # compute regression
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'num_classes': 1, # compute regression
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                }
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','dropouts','batch_size','learning_rate'],
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
            }
        }
        return tmp
    
    def __LMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate, 0 means don't use adaptive learning rate
                'seed': 1111,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'output_dim': 1,
                    'use_softmax': False,
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value
                },
                'sims':{
                    'output_dim': 1,
                    'use_softmax': False,
                    'grad_clip': 0.0, # gradient clip value
                },
            },
            'debugParas':{
                'd_paras': ['hidden_dims','dropouts','rank','batch_size','learning_rate','factor_lr','weight_decay'],
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'dropouts': random.choice([(0.3, 0.3, 0.3, 0.5),(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4)]),
                'rank': random.choice([3,4,5,6]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'factor_lr': random.choice([1e-4,5e-4,1e-3]), # factor_learning_rate
                'weight_decay': random.choice([1e-4,1e-3,5e-3]),
            }
        }
        return tmp
    
    def __MFN(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'output_dim': 1,
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'output_dim': 1,
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
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
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
            }
        }
        return tmp
    
    def __EF_LSTM(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'output_dim': 1,
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'output_dim': 1,
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
            'debugParas':{
                'd_paras': ['hidden_dims','num_layers','dropout','batch_size','learning_rate'],
                'hidden_dims': random.choice([16,32,64,128,256]),
                'num_layers': random.choice([2,3,4]),
                'dropout': random.choice([0.2,0.3,0.4,0.5]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
            }
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
                'seed': 1111,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','dropouts','batch_size','learning_rate'],
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4),(0.3, 0.3, 0.3, 0.5)]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
            }
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
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','post_text_dim','post_audio_dim',\
                            'post_video_dim','dropouts','post_dropouts','batch_size','learning_rate',\
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay'],
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
            }
        }
        return tmp

    def __MLMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate, 0 means don't use adaptive learning rate
            },
            'datasetParas':{
                'sims':{
                    'output_dim': 1,
                    'use_softmax': False,
                    'grad_clip': 0.0, # gradient clip value
                },
            },
            # dataset
            'debugParas':{
                'd_paras': ['hidden_dims','dropouts','rank','batch_size','learning_rate','factor_lr','weight_decay',\
                            'post_text_dim', 'post_audio_dim', 'post_video_dim', 'post_dropouts',\
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay'],
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([4,5]),
                'post_video_dim': random.choice([8,16,32,64]),
                'post_dropouts': random.choice([(0.2,0.2,0.2),(0.3,0.3,0.3),(0.4,0.4,0.4),(0.5,0.5,0.5)]),
                'dropouts': random.choice([(0.3, 0.3, 0.3, 0.5),(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4)]),
                'rank': random.choice([3,4,5,6]),
                # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'factor_lr': random.choice([1e-4,5e-4,1e-3]), # factor_learning_rate
                'weight_decay': random.choice([1e-4,1e-3,5e-3]),
                # dropout
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'text_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'audio_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'video_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
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
                'patience': 0, # when to decay learning rate
                # Logistics
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
            'debugParas':{
                'd_paras': ['hidden_dims','text_out','post_fusion_dim','post_text_dim','post_audio_dim',\
                            'post_video_dim','dropouts','post_dropouts','batch_size','learning_rate',
                            'M', 'T', 'A', 'V', 'text_weight_decay', 'audio_weight_decay', 'video_weight_decay'],
                'hidden_dims': random.choice([(128,16,128),(64,16,64),(128,32,128),(256,32,256),(64,32,64)]),
                'text_out': random.choice([32,64,128,256]),
                'post_fusion_dim': random.choice([16,32,64,128]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([4,5]),
                'post_video_dim': random.choice([8,16,32,64]),
                'dropouts': random.choice([(0.3,0.3,0.3),(0.2,0.2,0.2),(0.4,0.4,0.4),(0.3, 0.3, 0.3)]),
                'post_dropouts': random.choice([(0.2,0.2,0.2,0.2),(0.3,0.3,0.3,0.3),(0.4,0.4,0.4,0.4),(0.5,0.5,0.5,0.5)]),
                # # ref Original Paper
                'batch_size': random.choice([32,64,128]),
                'learning_rate': random.choice([5e-4,1e-3,2e-3,5e-3]),
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0,0.2,0.4,0.6,0.8,1]),
                'text_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'audio_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
                'video_weight_decay': random.choice([0, 1e-3, 1e-4, 1e-5]),
            }
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
                            **self.HYPER_MODEL_MAP[model_name]()['debugParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug_mode', type=bool, default=True,
                            help='if true, less data will be loaded')
        parser.add_argument('--modelName', type=str, default='lf_dnn',
                            help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn')
        parser.add_argument('--datasetName', type=str, default='mosi',
                            help='support mosi')
        parser.add_argument('--times', type=int, default=5,
                            help='how many times will be runned')
        parser.add_argument('--model_save_path', type=str, default='model_save',
                            help='path to save model.')
        parser.add_argument('--gpu_ids', type=list, default=[2],
                            help='indicates the gpus will be used.')
        return parser.parse_args()
        
    args = parse_args()
    config = ConfigDebug(args)
    args = config.get_config()
    print(args)