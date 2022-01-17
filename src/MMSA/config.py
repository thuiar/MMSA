import json
import os
import random
from easydict import EasyDict as edict


def get_config_regression(config_file, model_name, dataset_name):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): path to config file
        model_name (str): name of model
        dataset_name (str): name of dataset

    Returns:
        config (dict): config of the given dataset and model
    """
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_tune(config_file, model_name, dataset_name):
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    for item in model_debug_args['d_paras']:
        if type(model_debug_args[item]) == list:
            model_debug_args[item] = random.choice(model_debug_args[item])
        elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
            for k, v in model_debug_args[item].items():
                model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_debug_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    

    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_all(config_file):
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    return edict(config_all)


if __name__ == "__main__":
    config = get_config_tune("src/MMSA/config/config_tune.json", "mfm", "sims")
    print(config)