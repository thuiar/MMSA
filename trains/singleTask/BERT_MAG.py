import os
import time
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
# sys.path.insert(0,os.path.join(os.getcwd(), 'pytorch-transformers'))
from pytorch_transformers.optimization import AdamW

class BERT_MAG():
    def __init__(self, args):
        assert args.tasks in ['M']
        self.args = args
        # self.regression_criterion = nn.L1Loss()
        self.regression_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while(epochs - best_epoch < self.args.early_stop): 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].squeeze().to(self.args.device)
                    # forward
                    outputs = model(text, audio, vision)
                    logits = outputs[0].squeeze()
                    # compute loss
                    if self.args.output_mode == "classification":
                        loss = self.classification_criterion(logits.view(-1, self.args.num_labels), labels.view(-1))
                    elif self.args.output_mode == "regression":
                        loss = self.regression_criterion(logits.view(-1), labels.view(-1))
                    # backward
                    loss.backward()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(logits.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true, exclude_zero=self.args.excludeZero)
            print('%s: >> ' %(self.args.tasks) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.tasks[0]][self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= best_valid if min_or_max == 'min' else cur_valid >= best_valid
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_path = os.path.join(self.args.model_save_path,\
                                    f'{self.args.modelName}-{self.args.datasetName}-{self.args.tasks}.pth')
                if os.path.exists(model_path):
                    os.remove(model_path)
                # save model
                torch.save(model.cpu().state_dict(), model_path)
                model.to(self.args.device)
                print('save model in %s...' % model_path)
                self.do_test(model, dataloader['test'], mode="TEST")
            # early stop
            # if epochs - best_epoch >= self.args.early_stop:
            #     return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)
                    logits = outputs[0].squeeze()
                    if self.args.output_mode == "classification":
                        loss = self.classification_criterion(logits.view(-1, self.args.num_labels), labels.view(-1))
                    elif self.args.output_mode == "regression":
                        loss = self.regression_criterion(logits.view(-1), labels.view(-1))
                    eval_loss += loss.item()
                    y_pred.append(logits.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = self.metrics(pred, true, exclude_zero=self.args.excludeZero)
        print('%s-%s: >> ' %(mode, self.args.tasks) + dict_to_str(test_results))
        test_results['Loss'] = eval_loss
        tmp = {
            self.args.tasks: test_results
        }
        return tmp