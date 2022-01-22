import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
from multiprocessing import Pool
from pathlib import Path
from webbrowser import get

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from .config import get_config_regression, get_config_tune
from .data_loader import MMDataLoader
from .models import AMIO
from .trains import ATIO
from .utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # This is crucial for reproducibility


SUPPORTED_MODELS = ['lf_dnn', 'ef_lstm', 'tfn', 'lmf', 'mfn', 'graph_mfn', 'mult', 'misa', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm']
SUPPORTED_DATASETS = ['mosi', 'mosei', 'sims']

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
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=50, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1
):
    """
    Main function for running the MMSA framework.

    Parameters:
        model_name (str): Name of model
        dataset_name (str): Name of dataset
        config (dict): Config dict. Used to override arguments in config_file. Ignored in tune mode.
        config_file: Path to config file. If not specified, default config file will be used.
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
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(initial_args['device'])

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
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            random.seed(seeds[0])
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
            result = _run(args, num_workers, is_tune)
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
        args = get_config_regression(model_name, dataset_name, config_file)
        args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
        args['device'] = assign_gpu(gpu_ids)
        args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        if config: # override some arguments
            args.update(config)

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(args['device'])

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
            result = _run(args, num_workers, is_tune)
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


def _run(args, num_workers=4, is_tune=False, from_sena=False):
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
    epoch_results = trainer.do_train(model, dataloader)
    # epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    # load trained model & do test
    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    if from_sena:
        final_results = {}
        final_results['train'] = trainer.do_test(model, dataloader['train'], mode="TRAIN", return_sample_results=True)
        final_results['valid'] = trainer.do_test(model, dataloader['valid'], mode="VALID", return_sample_results=True)
        final_results['test'] = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
    elif is_tune:
        # use valid set to tune hyper parameters
        # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # delete saved model
        Path(args['model_save_path']).unlink(missing_ok=True)
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return {"epoch_results": epoch_results, 'final_results': final_results} if from_sena else results


SENA_ENABLED = True
try:
    import datetime
    from multiprocessing import Queue
    from sklearn.decomposition import PCA
    from flask_sqlalchemy import SQLAlchemy
except ImportError:
    SENA_ENABLED = False

if SENA_ENABLED:
    def SENA_run(
        db: SQLAlchemy, table: dict, task_id: int, progress_q: Queue,
        parameters: str, model_name: str, dataset_name: str, is_tune: bool,
        tune_times: int, feature_T: str, feature_A: str, feature_V: str,
        model_save_dir: str, res_save_dir: str, log_dir: str,
        gpu_ids: list, num_workers: int, seed: int, desc: str
    ) -> None:
        """
        Run M-SENA tasks. Should only be called from M-SENA Platform.
        Run only one seed at a time.

        Parameters:
            db (SQLAlchemy object): Used to store training results and task status.
            table (dict): Name and definitions of database tables.
            task_id (int): Task id.
            progress_q (multiprocessing.Queue): Used to communicate with M-SENA Platform.
            parameters (str): Training parameters in JSON.
            model_name (str): Model name.
            dataset_name (str): Dataset name.
            is_tune (bool): Whether to tune hyper parameters.
            tune_times (int): Number of times to tune hyper parameters.
            feature_T (str): Path to text feature file.
            feature_A (str): Path to audio feature file.
            feature_V (str): Path to video feature file.
            model_save_dir (str): Path to save trained model.
            res_save_dir (str): Path to save training results.
            log_dir (str): Path to save training logs.
            gpu_ids (list): GPU ids.
            num_workers (int): Number of workers.
            seed (int): Only one seed.
            desc (str): Description.
        """
        try:
            logger = logging.getLogger('app')
            logger.info(f"M-SENA Task {task_id} started.")
            cur_task = db.session.query(table['Task']).filter(table['Task'].task_id == task_id).first()
            # load training parameters
            if parameters == "": # use default config file
                if is_tune: # TODO
                    config_file = Path(__file__).parent / "config" / "config_tune.json"
                    args = get_config_tune(model_name, dataset_name, config_file)
                else:
                    config_file = Path(__file__).parent / "config" / "config_regression.json"
                    args = get_config_regression(model_name, dataset_name, config_file)
            else: # load from JSON
                args = json.loads(parameters)
                args['model_name'] = model_name
                args['dataset_name'] = dataset_name
            args['feature_T'] = feature_T
            args['feature_A'] = feature_A
            args['feature_V'] = feature_V
            args['device'] = assign_gpu(gpu_ids)
            args['cur_seed'] = 1 # the _run function need this to print log
            args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
            args = edict(args)
            # create folders
            Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            Path(res_save_dir).mkdir(parents=True, exist_ok=True)
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # create db record
            args_dump = args.copy()
            args_dump['device'] = str(args_dump['device'])
            custom_feature = False if (feature_A == "" and feature_V == "" and feature_T == "") else True
            db_result = table['Result'](
                dataset_name=dataset_name, model_name=model_name,
                is_tune=is_tune, args=json.dumps(args_dump), save_model_path='',
                loss_value=0.0, accuracy=0.0, f1=0.0, mae=0.0, corr=0.0,
                description=desc, custom_feature=custom_feature
            )
            db.session.add(db_result)
            db.session.flush()
            # result_id is allocated now, so model_save_path can be determined
            args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{db_result.result_id}.pth"
            db_result.save_model_path = args['model_save_path']
            # start training
            try:
                torch.cuda.set_device(args['device'])
                logger.info(f"Running with seed {seed}:")
                logger.info(f"Args:\n{args}")
                setup_seed(seed)
                # actual training
                results_dict = _run(args, num_workers, is_tune, from_sena=True)
                # db operations
                sample_dict = {}
                samples = db.session.query(table['Dsample']).filter_by(dataset_name=dataset_name).all()
                for sample in samples:
                    key = sample.video_id + '$_$' + sample.clip_id
                    sample_dict[key] = [sample.sample_id, sample.annotation]
                # update final results of test set
                db_result.loss_value = results_dict['final_results']['test']['Loss']
                db_result.accuracy = results_dict['final_results']['test']['Non0_acc_2']
                db_result.f1 = results_dict['final_results']['test']['Non0_F1_score']
                db_result.mae = results_dict['final_results']['test']['MAE']
                db_result.corr = results_dict['final_results']['test']['Corr']
                # save features
                if not is_tune:
                    logger.info("Running feature PCA ...")
                    features = {
                        k: results_dict['final_results'][k]['Features']
                        for k in ['train', 'valid', 'test']
                    }
                    all_features = {}
                    for select_modes in [['train', 'valid', 'test'], ['train', 'valid'], ['train', 'test'], \
                                        ['valid', 'test'], ['train'], ['valid'], ['test']]:
                        # create label index dict
                        # {"Negative": [1,2,5,...], "Positive": [...], ...}
                        cur_labels = []
                        for mode in select_modes:
                            cur_labels.extend(results_dict['final_results'][mode]['Labels'])
                        cur_labels = np.array(cur_labels)
                        label_index_dict = {}
                        for k, v in args['annotations'].items():
                            label_index_dict[k] = np.where(cur_labels == v)[0].tolist()
                        # handle features
                        cur_mode_features_2d = {}
                        cur_mode_features_3d = {}
                        for name in ['Feature_t', 'Feature_a', 'Feature_v', 'Feature_f']: 
                            cur_features = []
                            for mode in select_modes:
                                cur_features.append(features[mode][name])
                            cur_features = np.concatenate(cur_features, axis=0)
                            # PCA analysis
                            pca=PCA(n_components=3, whiten=True)
                            features_3d = pca.fit_transform(cur_features)
                            # split by labels
                            cur_mode_features_3d[name] = {}
                            for k, v in label_index_dict.items():
                                cur_mode_features_3d[name][k] = features_3d[v].tolist()
                            # PCA analysis
                            pca=PCA(n_components=2, whiten=True)
                            features_2d = pca.fit_transform(cur_features)
                            # split by labels
                            cur_mode_features_2d[name] = {}
                            for k, v in label_index_dict.items():
                                cur_mode_features_2d[name][k] = features_2d[v].tolist()
                        all_features['-'.join(select_modes)] = {'2D': cur_mode_features_2d, '3D': cur_mode_features_3d}
                    # save features
                    save_path = args.model_save_path.parent / (args.model_save_path.stem + '.pkl')
                    with open(save_path, 'wb') as fp:
                        pickle.dump(all_features, fp, protocol = 4)
                    logger.info(f'Feature saved at {save_path}.')
                # update sample results
                for mode in ['train', 'valid', 'test']:
                    final_results = results_dict['final_results'][mode]
                    for i, cur_id in enumerate(final_results["Ids"]):
                        payload = table['SResults'](
                            result_id=db_result.result_id,
                            sample_id=sample_dict[cur_id][0],
                            label_value=sample_dict[cur_id][1],
                            predict_value= 'Negative' if final_results["SResults"][i] < 0 else 'Positive',
                            predict_value_r = final_results["SResults"][i]
                        )
                        db.session.add(payload)
                # update epoch results
                cur_results = {}
                for mode in ['train', 'valid', 'test']:
                    cur_epoch_results = results_dict['final_results'][mode]
                    cur_results[mode] = {
                        "loss_value":cur_epoch_results["Loss"],
                        "accuracy":cur_epoch_results["Non0_acc_2"],
                        "f1":cur_epoch_results["Non0_F1_score"],
                        "mae":cur_epoch_results["MAE"],
                        "corr":cur_epoch_results["Corr"]
                    }
                payload = table['EResult'](
                    result_id=db_result.result_id,
                    epoch_num=-1,
                    results=json.dumps(cur_results)
                )
                db.session.add(payload)

                epoch_num = len(results_dict['epoch_results']['train'])
                for i in range(0, epoch_num):
                    cur_results = {}
                    for mode in ['train', 'valid', 'test']:
                        cur_epoch_results = results_dict['epoch_results'][mode][i]
                        cur_results[mode] = {
                            "loss_value":cur_epoch_results["Loss"],
                            "accuracy":cur_epoch_results["Non0_acc_2"],
                            "f1":cur_epoch_results["Non0_F1_score"],
                            "mae":cur_epoch_results["MAE"],
                            "corr":cur_epoch_results["Corr"]
                        }
                    payload = table['EResult'](
                        result_id=db_result.result_id,
                        epoch_num=i + 1,
                        results=json.dumps(cur_results)
                    )
                    db.session.add(payload)
                db.session.commit()
            except Exception as e:
                logger.exception(e)
                db.session.rollback()
                # TODO: remove saved features
                raise e

            cur_task.state = 1
        except Exception as e:
            logger.exception(e)
            cur_task.state = 2
        finally:
            cur_task.end_time = datetime.now()
            db.session.commit()
            logger.info(f"Task {task_id} Finished.")
            