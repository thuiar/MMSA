import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils.functions import dict_to_str
from ...utils.metricsTop import MetricsTop

logger = logging.getLogger('MMSA')

class MISA():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader):
        self.model = model
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.learning_rate)
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # using accumulated gradients
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # forward
                    outputs = model(text, audio, vision)['M']
                    # compute loss
                    cls_loss = self.criterion(outputs, labels)
                    diff_loss = self.get_diff_loss()
                    domain_loss = self.get_domain_loss()
                    recon_loss = self.get_recon_loss()
                    cmd_loss = self.get_cmd_loss()

                    if self.args.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss
                    
                    loss = cls_loss + \
                           self.args.diff_weight * diff_loss + \
                           self.args.sim_weight * similarity_loss + \
                           self.args.recon_weight * recon_loss
                    # backward
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
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
                    outputs = model(text, audio, vision)['M']
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
    
    def get_domain_loss(self,):

        if self.args.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.Model.domain_label_t
        domain_pred_v = self.model.Model.domain_label_v
        domain_pred_a = self.model.Model.domain_label_a

        # True domain labels
        domain_true_t = torch.LongTensor([0]*domain_pred_t.size(0)).to(self.device)
        domain_true_v = torch.LongTensor([1]*domain_pred_v.size(0)).to(self.device)
        domain_true_a = torch.LongTensor([2]*domain_pred_a.size(0)).to(self.device)

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.args.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.Model.utt_shared_a, self.model.Model.utt_shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self, ):

        shared_t = self.model.Model.utt_shared_t
        shared_v = self.model.Model.utt_shared_v
        shared_a = self.model.Model.utt_shared_a
        private_t = self.model.Model.utt_private_t
        private_v = self.model.Model.utt_private_v
        private_a = self.model.Model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.Model.utt_t_recon, self.model.Model.utt_t_orig)
        loss += self.loss_recon(self.model.Model.utt_v_recon, self.model.Model.utt_v_orig)
        loss += self.loss_recon(self.model.Model.utt_a_recon, self.model.Model.utt_a_orig)
        loss = loss/3.0
        return loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
