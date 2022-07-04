import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...utils import MetricsTop, dict_to_str

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence #

__all__ = ['MCTN']

logger = logging.getLogger('MMSA')

class MCTN():
    def __init__(self, args):
        self.args = args
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name) #########

    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.init_lr)###

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
            epoch_loss = 0.0
            self.model.train()
            y_pred, y_true=[],[]
            with tqdm(dataloader['train']) as td:
                for i_batch, batch_data in enumerate(td):

                    if self.args.dataset_name == "mosei":
                        if i_batch / len(dataloader) >= 0.5:
                            break

                    self.model.zero_grad()
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    batch_size = text.size(0)
                
                    loss, pred = self.model(text, audio, vision, labels, lengths = None) 

                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item() * batch_size
                    y_pred.append(pred.cpu())
                    y_true.append(labels.cpu())
            train_loss = epoch_loss / self.args.train_samples
            y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
        
            train_results = self.metrics(y_pred, y_true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >>  main loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            # self.optimizer.step(val_results['Loss'])    # Decay learning rate by validation loss
            print(val_results)
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
                    loss, outputs = self.model(text, audio, vision, labels, lengths = None)
                    
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.model_name, dict_to_str(eval_results)))
        return eval_results
