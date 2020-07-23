import os
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config_run import Config
from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % args.gpu_ids[0] if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    # start running
    # do train
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    pretrained_path = os.path.join(args.model_save_path,\
                                        f'{args.modelName}-{args.datasetName}-{args.tasks}.pth')
    assert os.path.exists(pretrained_path)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    # do test
    if args.debug_mode: 
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VAL")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    return results

def run_debug(seeds, debug_times=50):
    print('You are using DEBUG mode!')
    for i in range(debug_times):
        # cancel random seed
        args = parse_args()
        setup_seed(int(time.time()))
        config = ConfigDebug(args)
        args = config.get_config()
        # print debugging params
        print("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, debug_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                print(k, ':', v)
        print("#"*90)
        print('Start running %s...' %(args.modelName))
        results = []
        for j, seed in enumerate(seeds):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args)[args.tasks[0]])
        # save results to csv
        print('Start saving results...')
        if not os.path.exists(args.res_save_path):
            os.makedirs(args.res_save_path)
        # load resutls file
        save_file_path = os.path.join(args.res_save_path, \
                            args.datasetName + '-' + args.modelName + '-' + args.tasks + '-debug.csv')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))
        # save results
        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        print('Results are saved to %s...' %(save_file_path))

def run_normal(seeds):
    model_results = []
    # run results
    for i, seed in enumerate(seeds):
        args = parse_args()
        args.cur_time = i+1
        # load config
        config = Config(args)
        args = config.get_config()
        setup_seed(seed)
        args['seed'] = seed
        print('Start running %s...' %(args.modelName))
        # runnning
        test_results = run(args)
        # restore results
        model_results.append(test_results)
    # save results
    criterions = list(model_results[0][args.tasks[0]].keys())
    df = pd.DataFrame(columns=["Model"] + criterions)
    for m in args.tasks:
        res = [args.modelName+'-'+m]
        for c in criterions:
            values = [r[m][c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
    save_path = os.path.join(args.res_save_path, \
                    args.datasetName + '-' + args.modelName + '-' + args.tasks + '.csv')
    if not os.path.exists(args.res_save_path):
        os.makedirs(args.res_save_path)
    df.to_csv(save_path, index=None)
    print('Results are saved to %s...' %(save_path))
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='adjust parameters ?')
    parser.add_argument('--modelName', type=str, default='ef_lstm',
                        help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/mtfn/mlmf/mlf_dnn')
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

if __name__ == '__main__':
    seeds = [1, 12, 123, 1234, 12345]
    if parse_args().debug_mode:
        run_debug(seeds, debug_times=200)
    else:
        run_normal(seeds)