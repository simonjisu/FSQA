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

class RelationNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        """output_size = max_len*4 + 1 (has_relation) """
        super().__init__()
        self.output_size = output_size
        self.relation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size*4+1)
        )
    
    def forward(self, x):
        o = self.relation_net(x)
        has_relation = o[:, 0:1].squeeze(-1).contiguous()
        relations = o[:, 1:]
        s_target, e_target, s_subject, e_subject = map(lambda x: x.squeeze(-1).contiguous(), relations.split(self.output_size, dim=-1))
        return has_relation, s_target, e_target, s_subject, e_subject

class NLUModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        # self.hparams: model_path, intent_size, tags_size, max_len
        self.outputs_keys = ['tags', 'intent', 'has_relation', 's_target', 'e_target', 's_subject', 'e_subject']
        # Networks
        cfg = BertConfig()
        self.bert_ner = BertForTokenClassification.from_pretrained(self.hparams.model_path, num_labels=self.hparams.tags_size)
        self.bert_pooler = BertPooler(cfg)
        self.intent_network = nn.Linear(cfg.hidden_size, self.hparams.intent_size)
        self.relation_network = RelationNetwork(cfg.hidden_size, self.hparams.max_len)
        
        # losses
        if self.hparams.stage == 'train':
            self.losses = {
                'bce': nn.BCEWithLogitsLoss(),
                'ce': nn.CrossEntropyLoss()
            }
            # metrics
            self.metrics = nn.ModuleDict({
                'train_': self.create_metrics(prefix='train_'),
                'val_': self.create_metrics(prefix='val_')
            })
    def contiguous(self, x):
        return x.squeeze(-1).contiguous().type_as(x)

    def create_metrics(self, prefix='train_'):
        m = nn.ModuleDict()
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.Precision(), torchmetrics.Recall()])
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
        # intent
        return tags_logits.view(-1, self.hparams.tags_size)

    def _forward_intent(self, pooled_outputs):
        intent_logits = self.intent_network(pooled_outputs)
        return intent_logits

    def _forward_relation(self, pooled_outputs):
        # pooled_outputs: (B, max_len, 768)
        # has_relation: (B, )
        # s_target, e_target, s_subject, e_subject: (B, max_len)
        has_relation_logits, s_target_logits, e_target_logits, s_subject_logits, e_subject_logits = \
            self.relation_network(pooled_outputs)
        return has_relation_logits, s_target_logits, e_target_logits, s_subject_logits, e_subject_logits

    def forward(self, input_ids, token_type_ids, attention_mask):
        # tags
        last_hidden_state = self._forward_bert(input_ids, token_type_ids, attention_mask)
        tags_logits = self._forward_tags(last_hidden_state)

        # intent
        pooled_outputs = self.bert_pooler(last_hidden_state)
        intent_logits = self._forward_intent(pooled_outputs)
        # relation
        has_relation_logits, s_target_logits, e_target_logits, s_subject_logits, e_subject_logits = \
            self._forward_relation(pooled_outputs)

        return {
            'tags': tags_logits,                       # (B*max_len, tags_size)
            'intent': intent_logits,                   # (B, intent_size)
            'has_relation': has_relation_logits,       # (B, )
            's_target': s_target_logits,               # (B, max_len)
            'e_target': e_target_logits,               # (B, max_len)
            's_subject': s_subject_logits,             # (B, max_len)
            'e_subject': e_subject_logits              # (B, max_len)
        }

    def forward_all(self, batch, prefix='train_'):
        outputs = self.forward(
            input_ids=batch['input_ids'], 
            token_type_ids=batch['token_type_ids'], 
            attention_mask=batch['attention_mask'], 
        )
        s_target, e_target = map(self.contiguous, batch['target_relation'].split(1, dim=-1))
        s_subject, e_subject = map(self.contiguous, batch['subject_relation'].split(1, dim=-1))
        targets = {
            'tags': batch['tags'].view(-1),         # (B*max_len, )
            'intent': batch['intent'],              # (B, )
            'has_relation': batch['has_relation'],  # (B, )
            's_target': s_target,                   # (B, )
            'e_target': e_target,                   # (B, )
            's_subject': s_subject,                 # (B, )
            'e_subject': e_subject                  # (B, )
        }
        loss = self.cal_loss(outputs, targets)
        self.log(f'{prefix}loss', loss, 
            on_step=True, on_epoch=True, sync_dist=self.hparams.multigpu)
        # logging
        self.cal_metrics(outputs, targets, prefix=prefix)
        return loss

    def cal_loss(self, outputs, targets):
        has_relation_loss = self.losses['bce'](outputs['has_relation'], targets['has_relation'].float())

        tags_loss = self.losses['ce'](outputs['tags'], targets['tags'])
        intent_loss = self.losses['ce'](outputs['intent'], targets['intent'])
        s_target_loss = self.losses['ce'](outputs['s_target'], targets['s_target'])
        e_target_loss = self.losses['ce'](outputs['e_target'], targets['e_target'])
        s_subject_loss = self.losses['ce'](outputs['s_subject'], targets['s_subject'])
        e_subject_loss = self.losses['ce'](outputs['e_subject'], targets['e_subject'])

        return tags_loss + intent_loss + s_target_loss + e_target_loss + s_subject_loss + e_subject_loss + has_relation_loss

    def cal_metrics(self, outputs, targets, prefix='train_'):
        outputs_metrics = defaultdict()
        for k in self.outputs_keys:
            for k_sub, v in self.metrics[prefix][k](outputs[k], targets[k]).items():
                outputs_metrics[k_sub] = v
        self.log_dict(outputs_metrics)

    def training_step(self, batch, batch_idx):
        loss = self.forward_all(batch, prefix='train_')
        return loss

    def validation_step(self, batch, batch_idx):   
        loss = self.forward_all(batch, prefix='val_')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def predict(self, input_ids, token_type_ids, attention_mask):
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        predicts = self._predict_from_outputs(outputs)
        return predicts

    def _predict_from_outputs(self, outputs):
        predicts = {k: outputs[k].argmax(-1) for k in ['tags', 'intent', 's_target', 'e_target', 's_subject', 'e_subject']}
        predicts['has_relation'] = (outputs['has_relation'].sigmoid() >= 0.5).byte()
        return predicts

    # def _get_relation_inputs(self, last_hidden_state, relation):
    #     x = torch.stack([last_hidden_state[i, s:e].mean(0) for i, (s, e) in enumerate(relation)])
    #     return x 

    # def _forward_relation(self, last_hidden_state, target_relation, subject_relation):
    #     target_inputs = self._get_relation_inputs(last_hidden_state, target_relation)
    #     subject_inputs = self._get_relation_inputs(last_hidden_state, subject_relation)
    #     relation_inputs = torch.concat([last_hidden_state[:, 0], target_inputs, subject_inputs], dim=1)
    #     has_relation_logits, s_target_logits, e_target_logits, s_subject_logits, e_subject_logits = self.relation_network(relation_inputs)
    #     return has_relation_logits, s_target_logits, e_target_logits, s_subject_logits, e_subject_logits
