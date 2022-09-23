import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class MMIM():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def train_mmilb(self, dataloader):
        epoch_loss = 0.0
        self.model.train()

        with tqdm(dataloader['train']) as td:
            for i_batch, batch_data in enumerate(td):

                # for mosei we only use 50% dataset in stage 1
                if self.args.dataset_name == "mosei":
                    if i_batch / len(dataloader['train']) >= 0.5:
                        break

                self.model.zero_grad()
                text = batch_data['text'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                vision = batch_data['vision'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device)
                labels = labels.view(-1, 1)
                if not self.args.need_data_aligned:
                    audio_lengths = batch_data['audio_lengths']
                    vision_lengths = batch_data['vision_lengths']
                else:
                    audio_lengths, vision_lengths = 0, 0
                
                batch_size = text.size(0)

                results = self.model(text, (audio, audio_lengths), (vision, vision_lengths))

                loss = -results['lld']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer_mmilb.step()
                
                epoch_loss += loss.item() * batch_size
                
        return epoch_loss / self.args.train_samples

    def train_others(self, dataloader):
        epoch_loss, y_pred, y_true = 0.0,  [], []
            
        self.model.train()
        mem_pos_tv, mem_neg_tv, mem_pos_ta, mem_neg_ta = [], [], [], []
        
        if self.args.add_va:
            mem_pos_va, mem_neg_va = [], []

        with tqdm(dataloader['train']) as td:
            for i_batch, batch_data in enumerate(td):

                self.model.zero_grad()
                text = batch_data['text'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                vision = batch_data['vision'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device)
                labels = labels.view(-1, 1)
                if not self.args.need_data_aligned:
                    audio_lengths = batch_data['audio_lengths']
                    vision_lengths = batch_data['vision_lengths']
                else:
                    audio_lengths, vision_lengths = 0, 0
                
                batch_size = text.size(0)

                if i_batch >= self.args.mem_size:
                    mem = {'tv':{'pos':mem_pos_tv, 'neg':mem_neg_tv},
                            'ta':{'pos':mem_pos_ta, 'neg':mem_neg_ta},
                            'va': {'pos':mem_pos_va, 'neg':mem_neg_va} if self.args.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}

                results = self.model(text, (audio, audio_lengths), (vision, vision_lengths),
                                            y=labels, mem=mem)
                y_loss = self.criterion(results['M'], labels)

                if len(mem_pos_tv) < self.args.mem_size:
                    mem_pos_tv.append(results['pn_dic']['tv']['pos'].detach())
                    mem_neg_tv.append(results['pn_dic']['tv']['neg'].detach())
                    mem_pos_ta.append(results['pn_dic']['ta']['pos'].detach())
                    mem_neg_ta.append(results['pn_dic']['ta']['neg'].detach())
                    if self.args.add_va:
                        mem_pos_va.append(results['pn_dic']['va']['pos'].detach())
                        mem_neg_va.append(results['pn_dic']['va']['neg'].detach())
                
                else: # memory is full! replace the oldest with the newest data
                    oldest = i_batch % self.args.mem_size
                    mem_pos_tv[oldest] = results['pn_dic']['tv']['pos'].detach()
                    mem_neg_tv[oldest] = results['pn_dic']['tv']['neg'].detach()
                    mem_pos_ta[oldest] = results['pn_dic']['ta']['pos'].detach()
                    mem_neg_ta[oldest] = results['pn_dic']['ta']['neg'].detach()

                    if self.args.add_va:
                        mem_pos_va[oldest] = results['pn_dic']['va']['pos'].detach()
                        mem_neg_va[oldest] = results['pn_dic']['va']['neg'].detach()

                if self.args.contrast:
                    loss = y_loss + self.args.alpha * results['nce'] - self.args.beta * results['lld']
                else:
                    loss = y_loss

                if i_batch > self.args.mem_size:
                    loss -= self.args.beta * results['H']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer_main.step()
                y_pred.append(results['M'].cpu())
                y_true.append(labels.squeeze().cpu())
                epoch_loss += loss.item() * batch_size

        pred, truth = torch.cat(y_pred), torch.cat(y_true)

        return epoch_loss / self.args.train_samples, pred, truth

    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model

        mmilb_param = []
        main_param = []
        bert_param = []

        for name, p in model.named_parameters():
            # print(name)
            if p.requires_grad:
                if 'bert' in name:
                    bert_param.append(p)
                elif 'mi' in name:
                    mmilb_param.append(p)
                else: 
                    main_param.append(p)
            
            for p in (mmilb_param+main_param):
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p)

        self.optimizer_mmilb = getattr(torch.optim, self.args.optim)(
            mmilb_param, lr=self.args.lr_mmilb, weight_decay=self.args.weight_decay_mmilb)
        
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.lr_bert},
            {'params': main_param, 'weight_decay': self.args.weight_decay_main, 'lr': self.args.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.args.optim)(
            optimizer_main_group
        )

        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=self.args.when, factor=0.5, verbose=True)

        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            if self.args.contrast:
                train_loss_mmilb = self.train_mmilb(dataloader)
            train_loss_main, pred, truth = self.train_others(dataloader)
            
            train_results = self.metrics(pred, truth)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> mmilb loss: {round(train_loss_mmilb, 4)} main loss: {round(train_loss_main, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            self.scheduler_main.step(val_results['Loss'])    # Decay learning rate by validation loss
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

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    self.model.zero_grad()
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0
                    # we don't need lld and bound anymore
                    outputs = self.model(text, (audio, audio_lengths), 
                                            (vision, vision_lengths))['M']
                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.model_name, dict_to_str(eval_results)))
        return eval_results
