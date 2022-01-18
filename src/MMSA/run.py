import gc
import logging
import multiprocessing as mp
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from .config import get_config_regression, get_config_tune
from .data_loader import MMDataLoader
from .models import AMIO
from .trains import ATIO
from .utils.functions import assign_gpu, count_parameters, setup_seed

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
    tune_times=50, feature_T="", feature_A="", feature_V="",
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
        tune_times (int): Number of times to tune hyper parameters. Default: 50
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
    if model_save_dir == "": # use default model save dir
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "": # use default result save dir
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "": # use default log save dir
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    logger.info("======================================== Program Start ========================================")
    
    if is_tune: # run tune
        setup_seed(seeds[0])
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(config_file, model_name, dataset_name)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V
        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = [] # save used params
        csv_file = res_save_dir / f"{dataset_name}-{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            new_args = get_config_tune(config_file, model_name, dataset_name)
            args.update(new_args)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # check if this param has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            has_debuged.append(cur_param)
            # actual running
            result = _run(args, model_save_dir, gpu_ids, num_workers, is_tune)
            # save result to csv file
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
    else: # run normal
        args = get_config_regression(config_file, model_name, dataset_name)
        args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        logger.info("Running with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        res_save_dir = Path(res_save_dir) / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")
            # actual running
            result = _run(args, model_save_dir, gpu_ids, num_workers, is_tune)
            logger.info(f"Result for seed {seed}: {result}")
            model_results.append(result)
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
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)

    # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
    # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
    # It solves the bug of RNN always running on gpu 0.
    torch.cuda.set_device(args['device'])

    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
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
    time.sleep(1)

    return results

