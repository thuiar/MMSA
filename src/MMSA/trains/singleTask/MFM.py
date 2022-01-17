import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils.functions import dict_to_str
from ...utils.metricsTop import MetricsTop

logger = logging.getLogger('MMSA')

class MFM():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(model.parameters(), weight_decay=self.args.weight_decay)
        l1_loss = nn.L1Loss()
        l2_loss = nn.MSELoss()
        device = self.args.device
        l1_loss = l1_loss.to(device)
        l2_loss = l2_loss.to(device)
        
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    decoded,mmd_loss,missing_loss = model(text, audio, vision)
                    # compute loss
                    [x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
                    y_hat = y_hat.squeeze(1)
                    mmd_loss = self.args.lda_mmd * mmd_loss
                    x_l = text.permute(1,0,2)
                    x_a = audio.permute(1,0,2)
                    x_v = vision.permute(1,0,2)
                    gen_loss = self.args.lda_xl * l2_loss(x_l_hat,x_l) + self.args.lda_xa * l2_loss(x_a_hat,x_a) + self.args.lda_xv * l2_loss(x_v_hat,x_v)
                    disc_loss = l1_loss(y_hat, labels)
                    loss = disc_loss + gen_loss + mmd_loss + missing_loss
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(y_hat.cpu())
                    y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"Training {self.args.model_name} with seed {self.args.cur_seed}: [{epochs - best_epoch}/{epochs}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
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
        l1_loss = nn.L1Loss()
        l2_loss = nn.MSELoss()
        device = self.args.device
        l1_loss = l1_loss.to(device)
        l2_loss = l2_loss.to(device)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    decoded,mmd_loss,missing_loss = model(text, audio, vision)
                    [x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
                    y_hat = y_hat.squeeze(1)

                    eval_loss += l1_loss(y_hat, labels).item()

                    y_pred.append(y_hat.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.model_name, dict_to_str(eval_results)))
        return eval_results
