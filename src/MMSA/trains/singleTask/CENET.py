import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from ...utils import MetricsTop, dict_to_str
from transformers import BertTokenizer

logger = logging.getLogger('MMSA')

class CENET():
    def __init__(self, args):
        self.args = args
        self.args.max_grad_norm = 2
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained)
        self.criterion = nn.L1Loss()
    def do_train(self, model, dataloader,return_epoch_results=False):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        CE_params = ['CE']
        if return_epoch_results:
            epoch_results = {'train': [],'valid': [],'test': []}
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)  and not any(nd in n for nd in CE_params)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {"params": model.Model.bert.encoder.CE.parameters(), 'lr':self.args.learning_rate, "weight_decay": self.args.weight_decay},
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)  and not any(nd in n for nd in CE_params)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        epochs, best_epoch = 0, 0 
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            y_pred = []
            y_true = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for index,batch_data in enumerate(td):
                    loss = 0.0
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M']
                    labels = labels.to(self.args.device).view(-1, 1)
                    optimizer.zero_grad()
                    outputs = model(text,audio,vision)
                    logits = outputs[0]
                    loss += self.criterion(logits, labels)
                    loss.backward()
                    if self.args.max_grad_norm != -1.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    logits = logits.detach().cpu()
                    labels = labels.detach().cpu()
                    train_loss += loss.item()
                    y_pred.append(logits)
                    y_true.append(labels)
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d)>> loss: %.4f " % (self.args.model_name, \
                        epochs - best_epoch, epochs, train_loss))
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info('%s: >> ' %('Multimodal') + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path )
                model.to(self.args.device)
            # early stop
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = []
        y_true = []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    loss = 0.0
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M']
                    labels = labels.to(self.args.device).view(-1, 1)
                    outputs = model(text,audio,vision)
                    logits = outputs[0]
                    loss += self.criterion(logits, labels)
                    eval_loss += loss.item()
                    logits = logits.detach().cpu()
                    labels = labels.detach().cpu()
                    y_pred.append(logits)
                    y_true.append(labels)
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        logger.info('%s: >> ' %('Multimodal') + dict_to_str(results))
        eval_results = results
        eval_results['Loss'] = eval_loss
        return eval_results