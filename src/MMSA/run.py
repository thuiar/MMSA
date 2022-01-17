import os
import gc
import time

import logging
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

from .utils.functions import setup_seed, assign_gpu, count_parameters
from .models import AMIO
from .trains import ATIO
from .data_loader import MMDataLoader
from .config import get_config_regression, get_config_tune

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


# SUPPORTED_MODELS = ['lf_dnn', 'ef_lstm', 'tfn', 'lmf', 'mfn', 'graph_mfn', 'mult', 'misa', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm']
# SUPPORTED_DATASETS = ['mosi', 'mosei', 'sims']
logger = logging.getLogger('MMSA')


def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA') 
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def MMSA_run(
    model_name, dataset_name, config_file="", seeds=[], is_tune=False,
    feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, task_id=None, progress_q=None
):
    """
    Main function for running the MMSA framework.

    Parameters:
        model_name (str): Name of model
        dataset_name (str): Name of dataset
        config_file (str): Path to config file. If not specified, default config file will be used.
        seeds (list): List of seeds. Default: [1111, 1112, 1113, 1114, 1115]
        is_tune (bool): Whether to tune hyper parameters. Default: False
        feature_T (str): Path to text feature file. 
        feature_A (str): Path to audio feature file. 
        feature_V (str): Path to video feature file.
        model_save_dir (str): Path to save trained models. Default: "~/MMSA/saved_models"
        res_save_dir (str): Path to save csv results. Default: "~/MMSA/results"
        log_dir (str): Path to save log files. Default: "~/MMSA/logs"
        gpu_ids (list): Specify which gpus to use. If a empty list is supplied, will auto assign to the most memory-free gpu. Default: [0]
                        Currently only support single gpu.
        num_workers (int): Number of workers used to load data. Default: 4
        verbose_level (int): Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1

        task_id (int): Task id for M-SENA. 
        progress_q (multiprocessing.Queue): Multiprocessing queue for progress reporting with M-SENA. 
    """
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config_regression.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    logger.info("===================================== Program Start =====================================")
    
    if is_tune: # run tune
        args = get_config_tune(config_file, model_name, dataset_name)
        args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        logger.info("Tuning parameters with args:")
        logger.info(args)
        Path(res_save_dir).joinpath("tune").mkdir(parents=True, exist_ok=True)
        # TODO: run tune
    else: # run normal
        args = get_config_regression(config_file, model_name, dataset_name)
        args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        logger.info("Running with args:")
        logger.info(args)
        res_save_dir = Path(res_save_dir) / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            logger.info(f"Running with seed {seed} [{i + 1}/{len(seeds)}].")
            # actual running
            model_results.append(_run(args, model_save_dir, gpu_ids, num_workers, is_tune))
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, model_save_dir, gpu_ids, num_workers, is_tune):
    args['model_save_path'] = os.path.join(model_save_dir, f"{args['model_name']}-{args['dataset_name']}.pth")
    args['device'] = assign_gpu(gpu_ids)
    # add tmp tensor to increase the temporary consumption of GPU
    tmp_tensor = torch.zeros((100, 100)).to(args['device'])
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])
    del tmp_tensor

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # using multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = ATIO().getTrain(args)
    # do train
    trainer.do_train(model, dataloader)
    # load trained model & do test
    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    if is_tune:
        # use valid set to tune hyper parameters
        results = trainer.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    return results


"""
def run_tune(self, args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                f'{args.dataset_name}-{args.model_name}-{args.train_mode}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    for i in range(tune_times):
        # cancel random seed
        setup_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        print(args)
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.model_name, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.model_name))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111]):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' %(save_file_path))

def run_normal(self, args):
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else:
            config = ConfigClassification(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.model_name))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = self._run(args)
        # restore results
        model_results.append(test_results)
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.dataset_name}-{args.train_mode}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.model_name]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

"""
