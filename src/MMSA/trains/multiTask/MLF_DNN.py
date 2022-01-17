import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils.functions import dict_to_str
from ...utils.metricsTop import MetricsTop

logger = logging.getLogger('MMSA')

class MLF_DNN():
    def __init__(self, args):
        assert args.dataset_name == 'sims'

        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": self.args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": self.args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": self.args.video_weight_decay}],
                                lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
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
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
            train_loss = train_loss / len(dataloader['train'])

            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' %(m) + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.args.tasks[0]]
        eval_results['Loss'] = eval_loss
        return eval_results
