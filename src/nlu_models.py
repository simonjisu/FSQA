import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from collections import defaultdict
from transformers import BertConfig, BertForTokenClassification

import math
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class FocalLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int
    label_smoothing: float
    def __init__(self, alpha=1, gamma=2, ignore_index: int=-100, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class BertPooler(nn.Module):
    def __init__(self, config):
        """from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert/modeling_bert.py#L627"""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class NLUModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        self.outputs_keys = ['tags', 'intent']
        # Networks
        cfg = BertConfig()
        self.bert_ner = BertForTokenClassification.from_pretrained(self.hparams.model_path, num_labels=self.hparams.tags_size)
        self.bert_pooler = BertPooler(cfg)
        self.intent_network = nn.Linear(cfg.hidden_size, self.hparams.intent_size)
        
        # losses
        if self.hparams.loss_type == 'ce':
            self.intent_loss = nn.CrossEntropyLoss()
            self.tags_loss = nn.CrossEntropyLoss()
        elif self.hparams.loss_type == 'focal':
            self.intent_loss = FocalLoss(alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
            self.tags_loss = FocalLoss(alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        else:
            raise NotImplementedError('Loss is not implemented')                 
        # metrics
        self.metrics = nn.ModuleDict({
            'train_': self.create_metrics(prefix='train_'),
            'val_': self.create_metrics(prefix='val_'),
            'test_': self.create_metrics(prefix='test_')
        })
            
    def contiguous(self, x):
        return x.squeeze(-1).contiguous().type_as(x)

    def create_metrics(self, prefix='train_'):
        m = nn.ModuleDict()
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.F1()])
        for k in self.outputs_keys:
            m[k] = metrics.clone(prefix+k+'_')
        return m

    def _forward_bert(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_ner.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def _forward_tags(self, last_hidden_state):
        tags_outputs = self.bert_ner.dropout(last_hidden_state)
        tags_logits = self.bert_ner.classifier(tags_outputs)
        return tags_logits.view(-1, self.hparams.tags_size)

    def _forward_intent(self, pooled_outputs):
        intent_logits = self.intent_network(pooled_outputs)
        return intent_logits

    def forward(self, input_ids, token_type_ids, attention_mask):
        # tags
        last_hidden_state = self._forward_bert(input_ids, token_type_ids, attention_mask)
        tags_logits = self._forward_tags(last_hidden_state)

        # intent
        pooled_outputs = self.bert_pooler(last_hidden_state)
        intent_logits = self._forward_intent(pooled_outputs)

        return {
            'tags': tags_logits,       # (B*max_len, tags_size)
            'intent': intent_logits,   # (B, intent_size)
        }

    def forward_all(self, batch, prefix='train_'):
        outputs = self.forward(
            input_ids=batch['input_ids'], 
            token_type_ids=batch['token_type_ids'], 
            attention_mask=batch['attention_mask'], 
        )

        targets = {
            'tags': batch['tags'].view(-1),    # (B*max_len, )
            'intent': batch['intent'],         # (B, )
        }
        loss = self.cal_loss(outputs, targets)
        self.log(f'{prefix}loss', loss, 
            on_step=True, on_epoch=True, sync_dist=self.hparams.multigpu)
        # logging
        self.cal_metrics(outputs, targets, prefix=prefix)
        return {'loss': loss}

    def cal_loss(self, outputs, targets):
        tags_loss = self.tags_loss(outputs['tags'], targets['tags'])
        intent_loss = self.intent_loss(outputs['intent'], targets['intent'])
        return tags_loss + intent_loss

    def cal_metrics(self, outputs, targets, prefix='train_'):
        outputs_metrics = defaultdict()
        for k in self.outputs_keys:
            for k_sub, v in self.metrics[prefix][k](outputs[k], targets[k]).items():
                outputs_metrics[k_sub] = v

        self.log_dict(outputs_metrics, on_step=False, on_epoch=True, sync_dist=self.hparams.multigpu) 

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward_all(batch, prefix='train_')
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward_all(batch, prefix='val_')
        return loss_dict

    def test_step(self, batch, batch_idx):   
        loss_dict = self.forward_all(batch, prefix='test_')
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay_rate
        )
        # lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0.001)
        lr_schedulers = CosineAnnealingWarmUpRestarts(
            optimizer, 
            T_0=self.hparams.schedular_T_0, 
            T_mult=self.hparams.schedular_T_mult, 
            eta_max=self.hparams.schedular_eta_max, 
            T_up=self.hparams.schedular_T_up, 
            gamma=self.hparams.schedular_gamma
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_schedulers}

    def predict(self, input_ids, token_type_ids, attention_mask):
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        predicts = self._predict_from_outputs(outputs)
        return predicts

    def _predict_from_outputs(self, outputs):
        predicts = {k: outputs[k].argmax(-1) for k in ['tags', 'intent']} 
        return predicts