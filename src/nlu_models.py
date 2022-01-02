import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from collections import defaultdict
from transformers import BertConfig, BertForTokenClassification

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
        # self.hparams: model_path, intent_size, tags_size, max_len
        self.outputs_keys = ['tags', 'intent']
        # Networks
        cfg = BertConfig()
        self.bert_ner = BertForTokenClassification.from_pretrained(self.hparams.model_path, num_labels=self.hparams.tags_size)
        self.bert_pooler = BertPooler(cfg)
        self.intent_network = nn.Linear(cfg.hidden_size, self.hparams.intent_size)
        
        # losses   
        self.intent_loss = nn.CrossEntropyLoss()
        self.tags_loss = nn.CrossEntropyLoss()
                 
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
        return loss

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
        loss = self.forward_all(batch, prefix='train_')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_all(batch, prefix='val_')

    def test_step(self, batch, batch_idx):   
        loss = self.forward_all(batch, prefix='test_')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_schedulers, 'monitor': 'val_loss'}

    def predict(self, input_ids, token_type_ids, attention_mask):
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        predicts = self._predict_from_outputs(outputs)
        return predicts

    def _predict_from_outputs(self, outputs):
        predicts = {k: outputs[k].argmax(-1) for k in ['tags', 'intent']} 
        return predicts