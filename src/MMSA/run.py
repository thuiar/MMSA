import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

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


SUPPORTED_MODELS = [
    'LF_DNN', 'EF_LSTM', 'TFN', 'LMF', 'MFN', 'Graph_MFN', 'MFM',
    'MulT', 'MISA', 'BERT_MAG', 'MLF_DNN', 'MTFN', 'MLMF', 'Self_MM', 'MMIM'
]
SUPPORTED_DATASETS = ['MOSI', 'MOSEI', 'SIMS']

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
    model_name: str, dataset_name: str, config_file: str = None,
    config: dict = None, seeds: list = [], is_tune: bool = False,
    tune_times: int = 50, custom_feature: str = None, feature_T: str = None, 
    feature_A: str = None, feature_V: str = None, gpu_ids: list = [0],
    num_workers: int = 4, verbose_level: int = 1,
    model_save_dir: str = Path().home() / "MMSA" / "saved_models",
    res_save_dir: str = Path().home() / "MMSA" / "results",
    log_dir: str = Path().home() / "MMSA" / "logs",
):
    """Train and Test MSA models.

    Given a set of hyper-parameters(via config), will train models on training
    and validation set, then test on test set and report the results. If 
    `is_tune` is set, will accept lists as hyper-parameters and conduct a grid
    search to find the optimal values.

    Args:
        model_name: Name of MSA model.
        dataset_name: Name of MSA dataset.
        config_file: Path to config file. If not specified, default config
            files will be used.
        config: Config dict. Used to override arguments in config_file. 
        seeds: List of seeds. Default: [1111, 1112, 1113, 1114, 1115]
        is_tune: Tuning mode switch. Default: False
        tune_times: Sets of hyper parameters to tune. Default: 50
        custom_feature: Path to custom feature file. The custom feature should
            contain features of all three modalities. If only one modality has
            customized features, use `feature_*` below. 
        feature_T: Path to text feature file. Provide an empty string to use
            default BERT features. Default: ""
        feature_A: Path to audio feature file. Provide an empty string to use
            default features provided by dataset creators. Default: ""
        feature_V: Path to video feature file. Provide an empty string to use
            default features provided by dataset creators. Default: ""
        gpu_ids: GPUs to use. Will assign the most memory-free gpu if an empty
            list is provided. Default: [0]. Currently only supports single gpu.
        num_workers: Number of workers used to load data. Default: 4
        verbose_level: Verbose level of stdout. 0 for error, 1 for info, 2 for
            debug. Default: 1
        model_save_dir: Path to save trained model weights. Default: 
            "~/MMSA/saved_models"
        res_save_dir: Path to save csv results. Default: "~/MMSA/results"
        log_dir: Path to save log files. Default: "~/MMSA/logs"
    """
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if config_file is not None:
        config_file = Path(config_file)
    else: # use default config files
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config_regression.json"
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    if model_save_dir is None: # use default model save dir
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir is None: # use default result save dir
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir is None: # use default log save dir
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    logger.info("======================================== Program Start ========================================")
    
    if is_tune: # run tune
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['custom_feature'] = custom_feature
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
            args.update(new_args)
            if config:
                if config.get('model_name'):
                    assert(config['model_name'] == args['model_name'])
                args.update(config)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # check if this param has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            # actual running
            setup_seed(seeds[0])
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)
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
        args['custom_feature'] = custom_feature
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        if config: # override some arguments
            if config.get('model_name'):
                assert(config['model_name'] == args['model_name'])
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
    # epoch_results = trainer.do_train(model, dataloader)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
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


def MMSA_test(
    config: dict | str,
    weights_path: str,
    feature_path: str, 
    # seeds: list = [], 
    gpu_id: int = 0, 
):
    """Test MSA models on a single sample.

    Load weights and configs of a saved model, input pre-extracted
    features of a video, then get sentiment prediction results.

    Args:
        model_name: Name of MSA model.
        config: Config dict or path to config file. 
        weights_path: Pkl file path of saved model weights.
        feature_path: Pkl file path of pre-extracted features.
        gpu_id: Specify which gpu to use. Use cpu if value < 0.
    """
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, 'r') as f:
            args = json.load(f)
    elif type(config) == dict or type(config) == edict:
        args = config
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    args['train_mode'] = 'regression' # backward compatibility.

    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    args['device'] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1]]
    args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get('use_bert', None):
            if type(text := feature['text_bert']) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            if type(text := feature['text']) == np.ndarray:
                text = torch.from_numpy(text).float()
        if type(audio := feature['audio']) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        if type(vision := feature['vision']) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get('need_normalized', None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        # TODO: write a do_single_test function for each model in trains
        if args['model_name'] == 'self_mm' or args['model_name'] == 'mmim':
            output = model(text, (audio, torch.tensor(audio.shape[1]).unsqueeze(0)), (vision, torch.tensor(vision.shape[1]).unsqueeze(0)))
        elif args['model_name'] == 'tfr_net':
            input_mask = torch.tensor(feature['text_bert'][1]).unsqueeze(0).to(device)
            output, _ = model((text, text, None), (audio, audio, input_mask, None), (vision, vision, input_mask, None))
        else:
            output = model(text, audio, vision)
        if type(output) == dict:
            output = output['M']
    return output.cpu().detach().numpy()[0][0]
        

SENA_ENABLED = True
try:
    from datetime import datetime
    from multiprocessing import Queue

    import mysql.connector
    from sklearn.decomposition import PCA
except ImportError:
    logger.debug("SENA_run is not loaded due to missing dependencies. Ignore this if you are not using M-SENA Platform.")
    SENA_ENABLED = False

if SENA_ENABLED:
    def SENA_run(
        task_id: int, progress_q: Queue, db_url: str,
        parameters: str, model_name: str, dataset_name: str,
        is_tune: bool, tune_times: int,
        feature_T: str, feature_A: str, feature_V: str,
        model_save_dir: str, res_save_dir: str, log_dir: str,
        gpu_ids: list, num_workers: int, seed: int, desc: str
    ) -> None:
        """
        Run M-SENA tasks. Should only be called from M-SENA Platform.
        Run only one seed at a time.

        Parameters:
            task_id (int): Task id.
            progress_q (multiprocessing.Queue): Used to communicate with M-SENA Platform.
            db_url (str): Database url.
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
        # TODO: add progress report
        cursor = None
        try:
            logger = logging.getLogger('app')
            logger.info(f"M-SENA Task {task_id} started.")
            time.sleep(1) # make sure task status is committed by the parent process
            # get db parameters
            db_params = db_url.split('//')[1].split('@')[0].split(':')
            db_user = db_params[0]
            db_pass = db_params[1]
            db_params = db_url.split('//')[1].split('@')[1].split('/')
            db_host = db_params[0]
            db_name = db_params[1]
            # connect to db
            db = mysql.connector.connect(
                user=db_user, password=db_pass, host=db_host, database=db_name
            )
            cursor = db.cursor()
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
            # determine feature_dims
            if args['feature_T']:
                with open(args['feature_T'], 'rb') as f:
                    data_T = pickle.load(f)
                if 'use_bert' in args and args['use_bert']:
                    args['feature_dims'][0] = 768
                else:
                    args['feature_dims'][0] = data_T['valid']['text'].shape[2]
            if args['feature_A']:
                with open(args['feature_A'], 'rb') as f:
                    data_A = pickle.load(f)
                args['feature_dims'][1] = data_A['valid']['audio'].shape[2]
            if args['feature_V']:
                with open(args['feature_V'], 'rb') as f:
                    data_V = pickle.load(f)
                args['feature_dims'][2] = data_V['valid']['vision'].shape[2]
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
            cursor.execute(
                """
                    INSERT INTO Result (dataset_name, model_name, is_tune, custom_feature, created_at,
                     args, save_model_path, loss_value, accuracy, f1, mae, corr, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (dataset_name, model_name, is_tune, custom_feature, datetime.now(), json.dumps(args_dump), '', 0, 0, 0, 0, 0, desc)
            )
            result_id = cursor.lastrowid
            # result_id is allocated now, so model_save_path can be determined
            args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{result_id}.pth"
            cursor.execute(
                "UPDATE Result SET save_model_path = %s WHERE result_id = %s", (str(args['model_save_path']), result_id)
            )
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
                cursor2 = db.cursor(named_tuple=True)
                cursor2.execute("SELECT * FROM Dsample WHERE dataset_name=%s", (dataset_name,))
                samples = cursor2.fetchall()
                for sample in samples:
                    key = sample.video_id + '$_$' + sample.clip_id
                    sample_dict[key] = (sample.sample_id, sample.annotation)
                # update final results of test set
                cursor.execute(
                    """UPDATE Result SET loss_value = %s, accuracy = %s, f1 = %s,
                    mae = %s, corr = %s WHERE result_id = %s""",
                    (
                        results_dict['final_results']['test']['Loss'],
                        results_dict['final_results']['test']['Non0_acc_2'],
                        results_dict['final_results']['test']['Non0_F1_score'],
                        results_dict['final_results']['test']['MAE'],
                        results_dict['final_results']['test']['Corr'],
                        result_id
                    )
                )
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
                        label_index_dict['Negative'] = np.where(cur_labels < 0)[0].tolist()
                        label_index_dict['Neutral'] = np.where(cur_labels == 0)[0].tolist()
                        label_index_dict['Positive'] = np.where(cur_labels > 0)[0].tolist()
                        # handle features
                        cur_mode_features_2d = {}
                        cur_mode_features_3d = {}
                        for name in ['Feature_t', 'Feature_a', 'Feature_v', 'Feature_f']: 
                            cur_features = []
                            for mode in select_modes:
                                if name in features[mode]:
                                    cur_features.append(features[mode][name])
                            if cur_features != []:
                                cur_features = np.concatenate(cur_features, axis=0)
                            # PCA analysis
                            if len(cur_features) != 0:
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
                        cursor.execute(
                            """ INSERT INTO SResults (result_id, sample_id, label_value, predict_value, predict_value_r)
                            VALUES (%s, %s, %s, %s, %s)""",
                            (result_id, sample_dict[cur_id][0], sample_dict[cur_id][1],
                            'Negative' if final_results["SResults"][i] < 0 else 'Positive',
                            float(final_results["SResults"][i]))
                        )
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
                cursor.execute(
                    "INSERT INTO EResult (result_id, epoch_num, results) VALUES (%s, %s, %s)",
                    (result_id, -1, json.dumps(cur_results))
                )

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
                    cursor.execute(
                        "INSERT INTO EResult (result_id, epoch_num, results) VALUES (%s, %s, %s)",
                        (result_id, i+1, json.dumps(cur_results))
                    )
                db.commit()
                logger.info(f"Task {task_id} Finished.")
            except Exception as e:
                logger.exception(e)
                db.rollback()
                # TODO: remove saved features
                raise e
            cursor.execute("UPDATE Task SET state = %s WHERE task_id = %s", (1, task_id))
        except Exception as e:
            logger.exception(e)
            logger.error(f"Task {task_id} Error.")
            if cursor:
                cursor.execute("UPDATE Task SET state = %s WHERE task_id = %s", (2, task_id))
        finally:
            if cursor:
                cursor.execute("UPDATE Task SET end_time = %s WHERE task_id = %s", (datetime.now(), task_id))
                db.commit()


    def DEMO_run(db_url, feature_file, model_name, dataset_name, result_id, seed):

        db_params = db_url.split('//')[1].split('@')[0].split(':')
        db_user = db_params[0]
        db_pass = db_params[1]
        db_params = db_url.split('//')[1].split('@')[1].split('/')
        db_host = db_params[0]
        db_name = db_params[1]
        # connect to db
        db = mysql.connector.connect(
            user=db_user, password=db_pass, host=db_host, database=db_name
        )
        cursor2 = db.cursor(named_tuple=True)
        cursor2.execute(
            "SELECT * FROM Result WHERE result_id = %s", (result_id,)
        )
        result = cursor2.fetchone()
        save_model_path = result.save_model_path
        assert Path(save_model_path).exists(), f"pkl file {save_model_path} not found."
        result_args = json.loads(result.args)
        args = get_config_regression(model_name, dataset_name)
        args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        args['cur_seed'] = 1
        args.update(result_args)
        args['feature_T'] = feature_file
        args['feature_A'] = feature_file
        args['feature_V'] = feature_file
        # args['device'] = assign_gpu([])
        args['device'] = 'cpu'
        setup_seed(seed)
        model = AMIO(args).to(args['device'])
        model.load_state_dict(torch.load(save_model_path))
        model.to(args['device'])
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        feature_a = torch.Tensor(features['audio']).unsqueeze(0)
        feature_t = torch.Tensor(features['text']).unsqueeze(0)
        feature_v = torch.Tensor(features['video']).unsqueeze(0)
        if 'need_normalized' in args and args['need_normalized']:
            feature_a = torch.mean(feature_a, dim=1, keepdims=True)
            feature_v = torch.mean(feature_v, dim=1, keepdims=True)
        model.eval()
        with torch.no_grad():
            outputs = model(feature_t, feature_a, feature_v)
        predict = round(float(outputs['M'].cpu().detach().squeeze()), 3)
        return predict
