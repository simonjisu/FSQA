import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from collections import defaultdict
from transformers import BertConfig, BertModel, AdamW
# from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from pytorch_lightning.utilities import rank_zero_warn

import math
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler
from module_crf import CRF

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
    def __init__(self, weight=None, alpha=1, gamma=2, ignore_index: int=-100, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if weight is not None:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class NLUModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        self.outputs_keys = ['tags', 'intent']
        # Networks
        cfg = BertConfig()
        self.bert = BertModel.from_pretrained(self.hparams.model_path)
        self.linear_crf = nn.Linear(cfg.hidden_size, self.hparams.tags_size)
        self.crf = CRF(num_tags=self.hparams.tags_size, batch_first=True)
        self.intent_network = nn.Linear(cfg.hidden_size, self.hparams.intent_size)

        # if self.hparams.weight_dict is not None:
        #     tags_weight = [self.get_fn(i, 'tags') for i in range(self.hparams.tags_size)]
        #     tags_weight = torch.FloatTensor([1 - (c/sum(tags_weight)) for c in tags_weight])

        #     intent_weight = [self.get_fn(i, 'intent') for i in range(self.hparams.intent_size)]
        #     intent_weight = torch.FloatTensor([1 - (c/sum(intent_weight)) for c in intent_weight])
        # else:
        #     tags_weight = None
        #     intent_weight = None
        # # losses
        # if self.hparams.loss_type == 'ce':
        #     self.intent_loss = nn.CrossEntropyLoss(weight=intent_weight)
        #     self.tags_loss = nn.CrossEntropyLoss(weight=tags_weight)
        # elif self.hparams.loss_type == 'focal':
        #     self.intent_loss = FocalLoss(weight=intent_weight, alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        #     self.tags_loss = FocalLoss(weight=tags_weight, alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        # else:
        #     raise NotImplementedError('Loss is not implemented')
        if self.hparams.loss_type == 'ce': 
            self.intent_loss_function = nn.CrossEntropyLoss()
        else:
            self.intent_loss_function = FocalLoss(alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        # metrics
        self.metrics = nn.ModuleDict({
            'train_': self.create_metrics(prefix='train_'),
            'val_': self.create_metrics(prefix='val_'),
            'test_': self.create_metrics(prefix='test_')
        })

    # def get_fn(self, x, name):
    #     if self.hparams.weight_dict[name].get(str(x)):
    #         return self.hparams.weight_dict[name].get(str(x))
    #     else:
    #         return 0

    def contiguous(self, x):
        return x.squeeze(-1).contiguous().type_as(x)

    def create_metrics(self, prefix='train_'):
        m = nn.ModuleDict()
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.F1()])
        for k in self.outputs_keys:
            m[k] = metrics.clone(prefix+k+'_')
        return m

    def _forward_bert(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return outputs

    def _forward_tags(self, last_hidden_state, mask=None, tags=None):
        emissions = self.linear_crf(last_hidden_state)  # (B, T, tags_size)
        if tags is not None:
            tags_loss = -1*self.crf(emissions, tags=tags, mask=mask, reduction='mean')  # crf returns negative likelihood loss
            tags_prediction = torch.LongTensor(self.crf.decode(emissions)).to(emissions.device)
        else:
            tags_loss = None
            tags_prediction = torch.LongTensor(self.crf.decode(emissions)).to(emissions.device)
        return tags_loss, tags_prediction

    def _forward_intent(self, pooled_outputs, intent=None):
        intent_logits = self.intent_network(pooled_outputs)
        if intent is not None:
            intent_loss = self.intent_loss_function(intent_logits, intent)
            intent_prediction = intent_logits.argmax(-1)
        else:
            intent_loss = None
            intent_prediction = intent_logits.argmax(-1)
        return intent_loss, intent_prediction

    def forward(self, input_ids, token_type_ids, attention_mask, tags=None, intent=None):
        # bert
        outputs = self._forward_bert(input_ids, token_type_ids, attention_mask)
        # tags
        tags_loss, tags_prediction = self._forward_tags(
            last_hidden_state=outputs.last_hidden_state, 
            mask=attention_mask,
            tags=tags
        )
        # intent
        intent_loss, intent_prediction = self._forward_intent(
            pooled_outputs=outputs.pooler_output,
            intent=intent
        )

        return {
            'tags_loss': tags_loss,       # (B,)
            'intent_loss': intent_loss,      # (B,)
            'tags_pred': tags_prediction.view(-1),  # (B*T,)
            'intent_pred': intent_prediction,      # (B,)
        }

    def forward_all(self, batch, prefix='train_'):
        outputs = self(**batch)
        loss = self.cal_loss(outputs)
        self.log(f'{prefix}loss', loss, on_step=True, on_epoch=True, sync_dist=self.hparams.multigpu)
        
        return {
            'loss': loss, 
            'metric_tags': (outputs['tags_pred'], batch['tags'].view(-1)),    # (B*T, )
            'metric_intent': (outputs['intent_pred'], batch['intent']),    # (B, )
        }

    def cal_loss(self, outputs):
        # tags_loss = self.tags_loss(outputs['tags'], targets['tags'])
        # intent_loss = self.intent_loss(outputs['intent'], targets['intent'])
        tags_loss = outputs['tags_loss']
        intent_loss = outputs['intent_loss']
        return tags_loss + intent_loss

    def cal_metrics(self, preds, targets, prefix='train_'):
        outputs_metrics = defaultdict()
        for k in self.outputs_keys:
            for k_sub, v in self.metrics[prefix][k](preds[k], targets[k]).items():
                outputs_metrics[k_sub] = v

        self.log_dict(outputs_metrics, on_step=False, on_epoch=True, sync_dist=self.hparams.multigpu)
        
    def _preprocess_for_metrics(self, step_outputs):
        preds, targets = defaultdict(list), defaultdict(list)
        for o in step_outputs:
            preds['tags'].append(o['metric_tags'][0])
            preds['intent'].append(o['metric_intent'][0])
            targets['tags'].append(o['metric_tags'][1])
            targets['intent'].append(o['metric_intent'][1])

        for k in self.outputs_keys:
            preds[k] = torch.cat(preds[k])
            targets[k] = torch.cat(targets[k])
        return preds, targets

    def reset_metrics(self, prefix):
        for k in self.outputs_keys:
            self.metrics[prefix][k].reset()

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward_all(batch, prefix='train_')
        return loss_dict

    def training_epoch_end(self, step_outputs):
        preds, targets = self._preprocess_for_metrics(step_outputs)
        self.cal_metrics(preds, targets, prefix='train_')
        self.reset_metrics(prefix='train_')

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward_all(batch, prefix='val_')
        return loss_dict

    def validation_epoch_end(self, step_outputs):
        preds, targets = self._preprocess_for_metrics(step_outputs)
        self.cal_metrics(preds, targets, prefix='val_')
        self.reset_metrics(prefix='val_')

    def test_step(self, batch, batch_idx):   
        loss_dict = self.forward_all(batch, prefix='test_')
        return loss_dict

    def test_epoch_end(self, step_outputs):
        preds, targets = self._preprocess_for_metrics(step_outputs)
        self.cal_metrics(preds, targets, prefix='test_')
        self.reset_metrics(prefix='test_')

    def configure_optimizers(self):      
        params = list(self.bert.named_parameters()) + \
            list(self.linear_crf.named_parameters()) + \
            list(self.crf.named_parameters()) + \
            list(self.intent_network.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                'weight_decay': self.hparams.weight_decay_rate
            },
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        if self.hparams.optim_type == 'Adam':
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters, 
                lr=self.hparams.lr, 
                eps=self.hparams.optim_eps
            )
        elif self.hparams.optim_type == 'AdamW':
            optimizer = AdamW(
                optimizer_grouped_parameters, 
                lr=self.hparams.lr, 
                eps=self.hparams.optim_eps
            )
        else:
            raise NotImplementedError('No Optimizer')
        
        if self.hparams.schedular_type == 'cosine':
            lr_schedulers = torch.optim.lr_scheduler.LambdaLR(optimizer, self.cosine_with_restarts_lr_lambda, -1)
        elif self.hparams.schedular_type == 'cosine_with_restarts':
            lr_schedulers = torch.optim.lr_scheduler.LambdaLR(optimizer, self.cosine_lr_lambda, -1)
        else:
            raise NotImplementedError('No schedular')

        return {'optimizer': optimizer, 'lr_scheduler': lr_schedulers}

    def predict(self, input_ids, token_type_ids, attention_mask):
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        predicts = self._predict_from_outputs(outputs)
        return predicts

    def _predict_from_outputs(self, outputs):
        predicts = {k: outputs[k].argmax(-1) for k in ['tags', 'intent']} 
        return predicts

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.
        https://github.com/PyTorchLightning/pytorch-lightning/issues/10760
        """
        if self.trainer.num_training_batches != float('inf'):
            dataset_size = self.trainer.num_training_batches
        else:
            rank_zero_warn('Requesting dataloader...')
            dataset_size = len(self.trainer._data_connector._train_dataloader_source.dataloader())

        if isinstance(self.trainer.limit_train_batches, int):
            dataset_size = min(dataset_size, self.trainer.limit_train_batches)
        else:
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        accelerator_connector = self.trainer._accelerator_connector
        if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
            effective_devices = 1
        else:
            effective_devices = self.trainer.devices

        effective_devices = effective_devices * self.trainer.num_nodes
        effective_batch_size = self.trainer.accumulate_grad_batches * effective_devices
        max_estimated_steps = math.ceil(dataset_size // effective_batch_size) * self.trainer.max_epochs

        max_estimated_steps = min(max_estimated_steps, self.trainer.max_steps) if self.trainer.max_steps != -1 else max_estimated_steps
        return max_estimated_steps

    def cosine_with_restarts_lr_lambda(self, current_step):
        if current_step < self.hparams.schedular_warmup_steps:
            return float(current_step) / float(max(1, self.hparams.schedular_warmup_steps))
        progress = float(current_step - self.hparams.schedular_warmup_steps) / float(max(1, self.num_training_steps - self.hparams.schedular_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.hparams.schedular_num_cycles) * progress) % 1.0))))

    def cosine_lr_lambda(self, current_step):
        if current_step < self.hparams.schedular_warmup_steps:
            return float(current_step) / float(max(1, self.hparams.schedular_warmup_steps))
        progress = float(current_step - self.hparams.schedular_warmup_steps) / float(max(1, self.num_training_steps - self.hparams.schedular_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.hparams.schedular_num_cycles) * 2.0 * progress)))
