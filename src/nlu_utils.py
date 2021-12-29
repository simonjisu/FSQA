import pickle
import spacy
import torch
from pathlib import Path
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

class NLUTokenizer:
    def __init__(
        self, 
        hugg_path='bert-base-uncased', 
        spacy_path='en_core_web_sm'
    ):
        self.tokenizer = BertTokenizerFast.from_pretrained(hugg_path)
        self.spacy_nlp = spacy.load(spacy_path)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)


    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

    @classmethod
    def offsets_to_iob_tags(cls, encodes, ents, get_acc_relation=False):
        """
        ```
        IOB SCHEME
        I - Token is inside an entity.
        O - Token is outside an entity.
        B - Token is the beginning of an entity.

        BILUO SCHEME
        B - Token is the beginning of a multi-token entity.
        I - Token is inside a multi-token entity.
        L - Token is the last token of a multi-token entity.
        U - Token is a single-token unit entity.
        O - Token is outside an entity.
        ```
        method: IOB SCHEME
        modified from https://github.com/explosion/spaCy/blob/9d63dfacfc85e7cd6db7190bd742dfe240205de5/spacy/training/iob_utils.py#L63

        encodes: batch encodes from huggingface TokenizerFast
        ents: entities with start & end characters in sentences + entity
        """
        acc_relation = list()

        starts, ends = dict(), dict()
        for tkn_idx, (s_idx, e_idx) in enumerate(encodes['offset_mapping']):
            if s_idx == e_idx == 0:
                continue
            starts[s_idx] = tkn_idx
            ends[e_idx] = tkn_idx
        
        char_in_ents = {}
        labels = ['-'] * len(encodes['input_ids'])
        for s_char, e_char, ent in ents:
            if not ent:
                for s in starts:
                    labels[starts[s]] = 'O'
            else:
                for char_idx in range(s_char, e_char):
                    if char_idx in char_in_ents.keys():
                        raise ValueError(f'Trying to Overlapping same tokens: {char_in_ents[char_idx]} / {(s_char, e_char, ent)}')
                    char_in_ents[char_idx] = (s_char, e_char, ent)
                s_token = starts.get(s_char)
                e_token = ends.get(e_char)

                if s_token is not None and e_token is not None:
                    labels[s_token] = f'B-{ent}'
                    # add relation
                    if get_acc_relation and len(ent.split('.')) > 1:
                        acc_relation.append((s_token, e_token+1))

                    for i in range(s_token + 1, e_token+1):
                        labels[i] = f'I-{ent}'
                        
        entity_chars = set()
        for s_char, e_char, ent in ents:
            for i in range(s_char, e_char):
                entity_chars.add(i)
        for token_idx, (s, e) in enumerate(encodes['offset_mapping']):
            for i in range(s, e):
                if i in entity_chars:
                    break
            else:
                labels[token_idx] = 'O'
        if '-' in labels:
            raise ValueError('Some Tokens are not properly assigned' + f'{labels}')

        return labels, acc_relation

    def pad_tags(self, input_ids, tags, pad_idx:int=-100):
        padded_tags = [pad_idx] * len(input_ids)
        j = 0
        for i, tkn_id in enumerate(input_ids):
            if tkn_id in self.tokenizer.all_special_ids:
                continue
            padded_tags[i] = tags[j]
            j += 1
        return padded_tags

class NLUDataset(Dataset):
    def __init__(
        self, data, 
        tags2id=None, 
        intents2id=None, 
        hugg_path='bert-base-uncased', 
        spacy_path='en_core_web_sm', 
        max_len=128,
    ):
        self.questions, self.tags, self.intents, self.relations = list(zip(*data))
        self.tokenizer = NLUTokenizer(hugg_path, spacy_path)
        # question, entities, intent
        self.tags2id = tags2id
        self.intents2id = intents2id
        self.max_len = max_len

    def __getitem__(self, index):
        question = self.questions[index]
        tags = list(map(self.tags2id.get, self.tags[index]))
        intent = self.intents2id.get(self.intents[index])
        relation = self.relations[index]

        encodes = self.tokenizer(
            question, 
            return_offsets_mapping=False,
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
        )
        # labels = intent + tags
        tags = self.tokenizer.pad_tags(
            input_ids=encodes['input_ids'], 
            tags=tags, 
            pad_idx=0,
        )

        item = {k: torch.as_tensor(v) for k, v in encodes.items()}

        item['intent'] =  torch.as_tensor(intent)
        item['tags'] = torch.as_tensor(tags)
        item['has_relation'] = torch.as_tensor(relation[0])
        item['target_relation'] = torch.as_tensor(relation[1])
        item['subject_relation'] = torch.as_tensor(relation[2])
        return item

    def __len__(self):
        return len(self.questions)


class NLUDataModule(pl.LightningDataModule):
    def __init__(
        self, data_path:Path, ids_path:Path,
        batch_size:int=32, 
        max_len:int=128,
        test_size=0.1,
        num_workers=4,
        seed=777
    ):
        super().__init__()
        self.data_path = data_path
        self.ids_path = ids_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.test_size = test_size
        self.seed = seed
        self.num_workers = num_workers
        
    def load_data(self):
        with Path(self.data_path).open('rb') as file:
            data = pickle.load(file)
        
        with Path(self.ids_path).open('rb') as file:
            ids = pickle.load(file)

        self.train_data = data['train']
        self.test_data = data['test']
        self.tags2id = ids['tags2id']
        self.intents2id = ids['intents2id']

    def prepare_data(self):
        self.load_data()

    def train_dataloader(self):
        train_dataset = NLUDataset(
            self.train_data, 
            tags2id=self.tags2id, 
            intents2id=self.intents2id,
            max_len=self.max_len
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = NLUDataset(
            self.test_data, 
            tags2id=self.tags2id, 
            intents2id=self.intents2id,
            max_len=self.max_len
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)