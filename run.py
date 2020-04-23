import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.log import *
from utils.metricsTop import *
from utils.optimizerTop import *
from utils.lossTop import *
from config.config_run import *
from config.config_debug import *
from models.AIO import Terminator
from data.load_data import MMDataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def do_train(args, model, dataloader, criterion, scheduler, optimizer, metrics, device):
    # initilize results
    best_acc = 0
    epochs, best_epoch = 0, 0
    # loop util earlystop
    while True: 
        epochs += 1
        # train
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        losses = []
        model.train()
        train_loss = 0.0
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                vision = batch_data['vision'].to(device)
                audio = batch_data['audio'].to(device)
                text = batch_data['text'].to(device)
                labels = batch_data['labels']
                for k in labels.keys():
                    labels[k] = labels[k].to(device).view(-1, 1)
                # clear gradient
                optimizer.zero_grad()
                # forward
                outputs = model(text, audio, vision)
                # compute loss
                loss = criterion(outputs, labels)
                # backward
                loss.backward()
                if 'grad_clip' in args and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # update
                optimizer.step()
                # store results
                train_loss += loss.item()
                for m in args.modality:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(labels['M'].cpu())
        train_loss = train_loss / len(dataloader['train'])
        print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (args.modelName, \
                    epochs - best_epoch, epochs, args.cur_time, train_loss))
        for m in args.modality:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            train_results = metrics(pred, true)
            print('%s: >> ' %(m) + dict_to_str(train_results))
        # validation
        val_results = do_test(args, model, dataloader['valid'], criterion, metrics, device, mode="VAL")
        val_acc = val_results[args.modality[0]]['Mult_acc_2']
        if args.patience > 0:
            scheduler.step(val_acc)
        # save best model
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epochs
            old_models = glob(os.path.join(args.model_save_path,\
                                            f'{args.modelName}-{args.datasetName}-{args.modality}-*.pth'))
            for old_model_path in old_models:
                os.remove(old_model_path)
            # save model
            new_model_path = os.path.join(args.model_save_path,\
                                            f'{args.modelName}-{args.datasetName}-{args.modality}-val_acc-{val_acc:.4f}.pth')
            torch.save(model.cpu().state_dict(), new_model_path)
            model.to(device)
        # early stop
        if epochs - best_epoch >= args.early_stop:
            return

def do_test(args, model, dataloader, criterion, metrics, device, mode="VAL"):
    model.eval()
    y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
    y_true = {'M': [], 'T': [], 'A': [], 'V': []}
    eval_loss = 0.0
    with torch.no_grad():
        with tqdm(dataloader) as td:
            for batch_data in td:
                vision = batch_data['vision'].to(device)
                audio = batch_data['audio'].to(device)
                text = batch_data['text'].to(device)
                labels = batch_data['labels']
                for k in labels.keys():
                    labels[k] = labels[k].to(device).view(-1, 1)
                outputs = model(text, audio, vision)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                for m in args.modality:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(labels['M'].cpu())
    eval_loss = eval_loss / len(dataloader)
    print(mode+"-(%s)" % args.modelName + " >> loss: %.4f " % eval_loss)
    return_res = {}
    for m in args.modality:
        pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
        results = metrics(pred, true)
        print('%s: >> ' %(m) + dict_to_str(results))
        return_res[m] = results
    return return_res

def run(args):
    assert not ('M' in args.modality and args.modality[0] != 'M')
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # data
    dataloader = MMDataLoader(args)
    model = Terminator(args)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    status("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % args.gpu_ids[0] if using_cuda else 'cpu')
    model.to(device)
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    # optim and loss
    optimizer = OptimizerTop().getOptim(model, args)
    criterion = LossTop(args).getLoss()
    metrics = MetricsTop().getMetics(args.datasetName)
    # set adaptive learning rate
    if args.patience > 0:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True, patience=args.patience)
    else:
        scheduler = None
    # start running
    # do train
    do_train(args, model, dataloader, criterion, scheduler, optimizer, metrics, device)
    # load best model
    model_save_pathes = glob(os.path.join(args.model_save_path,\
                                            f'{args.modelName}-{args.datasetName}-{args.modality}-*.pth'))
    assert len(model_save_pathes) == 1
    model.load_state_dict(torch.load(model_save_pathes[0]))
    model.to(device)
    # do test
    results = do_test(args, model, dataloader['test'], criterion, metrics, device, mode="TEST")
    return results

def run_debug(seeds, debug_times=50):
    status('You are using DEBUG mode!')
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
        status('Start running %s...' %(args.modelName))
        results = []
        for j, seed in enumerate(seeds):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args)['M'])
        # save results to csv
        status('Start saving results...')
        results_dir = os.path.join(args.res_save_path,args.datasetName+'-debug')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_file_path = os.path.join(results_dir, args.modelName+'.csv')
        df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()]) if i==0 else pd.read_csv(save_file_path)
        tmp = []
        for col in list(df.columns):
            if col in args.d_paras:
                tmp.append(args[col])
            elif col in results[0].keys():
                values = [r[col] for r in results]
                tmp.append(round(sum(values) * 100 / len(values), 2))
        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        status('Results are saved to %s...' %(save_file_path))

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
        status('Start running %s...' %(args.modelName))
        # runnning
        test_results = run(args)
        # restore results
        model_results.append(test_results)
    # save results
    ms = args.modality # MTAV
    criterions = list(model_results[0][ms[0]].keys())
    df = pd.DataFrame(columns=["Model"] + criterions)
    for m in ms:
        res = [args.modelName+'-'+m]
        for c in criterions:
            values = [r[m][c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
    save_path = os.path.join(args.res_save_path, args.datasetName, \
                                args.modelName + '-' + args.modality + '.csv')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path, index=None)
    status('Results are saved to %s...' %(save_path))
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='debug parameters ?')
    parser.add_argument('--modelName', type=str, default='tfn',
                        help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/mtfn/mlmf/mlf_dnn')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/sims')
    parser.add_argument('--modality', type=str, default='M',
                        help='the tasks will be run')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='models',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

if __name__ == '__main__':
    seeds = [1, 12, 123, 1234, 12345]
    if parse_args().debug_mode:
        run_debug(seeds, debug_times=200)
    else:
        run_normal(seeds)