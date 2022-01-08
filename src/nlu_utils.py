import spacy
from spacy.tokens import token
import torch
import json
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union
from spacy.symbols import ORTH, NORM
from spacy.training import iob_to_biluo, biluo_to_iob, offsets_to_biluo_tags, biluo_tags_to_spans, biluo_tags_to_offsets
from collections import defaultdict
from tokenizations import get_alignments

def load_jsonl(path):
    with path.open('r', encoding='utf-8') as file:
        data = file.readlines()
        all_data = []
        for line in tqdm(data, total=len(data), desc='loading'):
            all_data.append(json.loads(line))
    return all_data

class NLUTokenizer:
    def __init__(
        self, 
        hugg_path='bert-base-uncased', 
        spacy_path='en_core_web_sm'
    ):
        self.bert = BertTokenizerFast.from_pretrained(hugg_path)
        self.spacy_nlp = spacy.load(spacy_path)
        for tkn in self.bert.all_special_tokens:
            self.spacy_nlp.tokenizer.add_special_case(tkn, [{ORTH: tkn, NORM: tkn}])

    def bert_tokenize(self, text):
        return self.bert.tokenize(text)

    def bert_decode(self, token_ids, **kwargs):
        return self.bert.decode(token_ids, **kwargs)

    def __call__(self, text, **kwargs):
        return self.bert(text, **kwargs)

    def spacy_tokenize(self, text):
        return [str(tkn) for tkn in self.spacy_nlp(text)]

    def get_tags(self, text, ents, tag_type='iob'):
        """
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
        """
        doc = self.spacy_nlp(text)
        biluo_tags = offsets_to_biluo_tags(doc, ents)
        if tag_type == 'iob':
            return biluo_to_iob(biluo_tags)
        elif tag_type == 'biluo':
            return biluo_tags
        else:
            raise KeyError(f"tag_type must be either `iob` or `biluo`, your is `{tag_type}`")

    def bert_tag_alignment(self, bert_tkns, spacy_tkns, tags):
        """
        mapped to spacy tags
        """
        biluo_tags = iob_to_biluo(tags)
        _, b2a = get_alignments(bert_tkns, spacy_tkns)
        mapped_tags = []
        for i, tkn in enumerate(spacy_tkns):
            mapped_tkn_ids = b2a[i]
            ts = [biluo_tags[j] for j in mapped_tkn_ids]
            mapped_tags.append(ts[0])
        return mapped_tags

    def spacy_lemma(self, text):
        return ' '.join([x.lemma_ for x in self.spacy_nlp(text)])

    def get_spacy_doc_entities(self, text, tags):
        bert_tkns = [s.lstrip('##') for s in self.bert_tokenize(text)]
        spacy_tkns = self.spacy_tokenize(text)
        mapped_tags = self.bert_tag_alignment(bert_tkns, spacy_tkns, tags)
        doc = self.spacy_nlp(text)
        # fix if we miss somethin
        # bert_ents = [('O', '') if t == 'O' else t.split('-') for t in mapped_tags]
        # spacy_ents = iob_to_biluo([x.ent_iob_ if x.ent_iob_ == 'O' else f'{x.ent_iob_}-{x.ent_type_}' for x in doc])
        # spacy_ents = [x.replace('DATE', 'TIME') for x in spacy_ents]
        # spacy_ents = [('O', '') if t == 'O' else t.split('-') for t in spacy_ents]
        # for i, (b_ent, s_ent) in enumerate(zip(bert_ents, spacy_ents)):
        #     if b_ent[1] == '' and s_ent[1] != '':
        #         bert_ents[i] = s_ent
        #     elif b_ent[1] == s_ent[1] and b_ent[0] != s_ent[0]:
        #         bert_ents[i] = s_ent
        # ensembled_tags = list(map(lambda x: x[0] if x[0] == 'O' else '-'.join(x), bert_ents))
        ensembled_tags = mapped_tags
        doc.ents = biluo_tags_to_spans(doc, ensembled_tags)
        entities = []
        for s, e, ent in biluo_tags_to_offsets(doc, ensembled_tags):
            lemma = self.spacy_lemma(text[s:e])
            entities.append((lemma, ent.upper()))

        return entities, doc

    def fix_tags_alignment(self, longer_tokens, shorter_tokens, tags):
        """
        return biluo tags, tags are mapped to longer tokens
        """
        a2b, _ = get_alignments(a=shorter_tokens, b=longer_tokens)
        biluo_tags = iob_to_biluo(tags)
        mapped_tags = ['-'] * len(longer_tokens)
        for i, tag in enumerate(biluo_tags):
            if tag == 'O':
                for k in a2b[i]:
                    mapped_tags[k] = tag
                continue

            prefix, label = tag.split('-')
            if prefix == 'B':
                for j, k in enumerate(a2b[i]):
                    if j == 0:
                        mapped_tags[k] = tag
                    else:
                        mapped_tags[k] = f'I-{label}'
            elif prefix == 'L':
                for j, k in enumerate(a2b[i]):
                    if j == len(a2b[i])-1:
                        mapped_tags[k] = tag
                    else:
                        mapped_tags[k] = f'I-{label}'
            elif prefix == 'U':
                if len(a2b[i]) == 1:
                    k = a2b[i][0]
                    mapped_tags[k] = tag
                elif len(a2b[i]) == 2:
                    b, l = a2b[i]
                    mapped_tags[b] = f'B-{label}'
                    mapped_tags[l] = f'L-{label}'
                else:
                    for j, k in enumerate(a2b[i]):
                        if j == 0:
                            mapped_tags[k] = f'B-{label}'
                        elif j == len(a2b[i])-1:
                            mapped_tags[k] = f'L-{label}'
                        else:
                            mapped_tags[k] = f'I-{label}'
            else:
                for j, k in enumerate(a2b[i]):
                    mapped_tags[k] = tag
        return mapped_tags

    def get_token_mappings(self, longer_tokens, shorter_tokens):
        """
        longer_tokens: spanned tokens
        shorter_tokens: origin tokens
        """
        i, j = 0, 0
        token_mappings = defaultdict(list) #{shorter_token: [longer_token]}
        spanned = ''
        while i < len(shorter_tokens) and j < len(longer_tokens):
            s_tkn = shorter_tokens[i]
            l_tkn = longer_tokens[j]
            if s_tkn == l_tkn:
                token_mappings[i].append(j)
                i += 1
                j += 1
                spanned = ''
            else:
                token_mappings[i].append(j)
                j += 1
                spanned += l_tkn[2:] if l_tkn.startswith('##') else l_tkn
                # see whether spanned is equal to current tokens
                if len(spanned) == len(s_tkn):
                    i += 1 
                    spanned = ''
        return token_mappings

    def map_spanned_tokens(self, longer_tokens, shorter_token, tags):
        """
        longer_tokens: spanned tokens (a. bert_tokens, b. spacy_tokens)
        shorter_tokens: not spanned tokens (a. spacy_tokens, b. original_tokens) 
        """
        token_mappings = self.get_token_mappings(longer_tokens, shorter_token)

        spanned_tags = ['-'] * len(longer_tokens)
        for i, t in enumerate(tags):
            for k in token_mappings[i]:
                spanned_tags[k] = t
        
        if "-" in spanned_tags:
            raise ValueError(f"problems in mapping: \n{spanned_tags}\n short: {shorter_token}\n Long: {longer_tokens}")
        return spanned_tags

    def offset_mapping_to_tags(self, offset_mapping, ents):
        starts, ends = dict(), dict()
        for tkn_idx, (s_idx, e_idx) in enumerate(offset_mapping):
            if s_idx == e_idx == 0:
                continue
            starts[s_idx] = tkn_idx
            ends[e_idx] = tkn_idx

        char_in_ents = {}
        labels = ['-'] * len(offset_mapping)
        for s_char, e_char, ent in ents:
            if not ent:
                for s in starts:  # account for many-to-one
                    if s >= s_char and s < e_char:
                        labels[starts[s]] = 'O'
            else:
                for char_idx in range(s_char, e_char):
                    if char_idx in char_in_ents.keys():
                        raise ValueError(f'Trying to Overlapping same tokens: {char_in_ents[char_idx]} / {(s_char, e_char, ent)}')
                    char_in_ents[char_idx] = (s_char, e_char, ent)
                s_token = starts.get(s_char)
                e_token = ends.get(e_char)

                if s_token is not None and e_token is not None:
                    if s_token == e_token:
                        labels[s_token] = f"U-{ent}"
                    else:
                        labels[s_token] = f"B-{ent}"
                        for i in range(s_token + 1, e_token):
                            labels[i] = f"I-{ent}"
                        labels[e_token] = f"L-{ent}"
                        
        entity_chars = set()
        for s_char, e_char, ent in ents:
            for i in range(s_char, e_char):
                entity_chars.add(i)
        for token_idx, (s, e) in enumerate(offset_mapping):
            for i in range(s, e):
                if i in entity_chars:
                    break
            else:
                labels[token_idx] = 'O'
        if '-' in labels:
            print(labels.index('-'))
            raise ValueError('Some Tokens are not properly assigned' + f'{labels}')

        return labels

    def pad_tags(self, tags):
        # padded_tags = [self.bert.cls_token_id] + tags + [self.bert.sep_token_id]
        padded_tags = [self.bert.pad_token_id] + tags + [self.bert.pad_token_id]
        return padded_tags

class NLUDataset(Dataset):
    def __init__(
        self, data, tags2id, intents2id,
        hugg_path='bert-base-uncased', 
        spacy_path='en_core_web_sm', 
        max_len=64,
        tag_type='iob'
    ):
        super().__init__()
        self.data = data
        self.tokenizer = NLUTokenizer(hugg_path, spacy_path)
        # question, entities, intent
        self.tags2id = tags2id
        self.intents2id = intents2id
        self.max_len = max_len
        self.tag_type = tag_type

    def __getitem__(self, index):
        text = self.data[index]['text']
        ents = self.data[index]['entities']
        intent = self.data[index]['intent']

        bert_offset_mapping = self.tokenizer.bert(
            text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        tags = self.tokenizer.offset_mapping_to_tags(offset_mapping=bert_offset_mapping, ents=ents)
        tags = biluo_to_iob(tags)
        
        bert_encodes = self.tokenizer(
            text, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.max_len
        )
        numeric_tags = list(map(self.tags2id.get, tags))
        padded_tags = self.tokenizer.pad_tags(numeric_tags)
        intent = self.intents2id.get(intent)
        
        item = {k: v for k, v in bert_encodes.items()}
        item['intent'] = intent
        item['tags'] = padded_tags
        return item

    def __len__(self):
        return len(self.data)

    @classmethod
    def custom_collate_fn(cls, items):
        # items = [{'A': 0, 'B': 1}, {'A': 100, 'B': 100}]
        # do padding
        max_len = max([len(item['input_ids']) for item in items])
        pad_idx = 0
        batch = defaultdict(list)
        for item in items:
            batch['input_ids'].append(item['input_ids'] + [pad_idx]*(max_len - len(item['input_ids'])))
            batch['token_type_ids'].append(item['token_type_ids'] + [pad_idx]*(max_len - len(item['token_type_ids'])))
            batch['attention_mask'].append(item['attention_mask'] + [pad_idx]*(max_len - len(item['attention_mask'])))
            batch['tags'].append(item['tags'] + [pad_idx]*(max_len - len(item['tags'])))
            batch['intent'].append(item['intent'])
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}
        return batch

class NLUDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_path:Path, 
        valid_path:Union[Path, None],
        test_path:Union[Path, None],
        labels_path:Path,
        batch_size:int=32, 
        max_len:int=64,
        num_workers=4,
        seed=777
    ):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.labels_path = labels_path

        with Path(self.labels_path).open('r', encoding='utf-8') as file:
            labels = json.load(file)

        self.tags2id = labels['tags']
        self.intents2id = labels['intent']

        self.batch_size = batch_size
        self.max_len = max_len
        self.seed = seed
        self.num_workers = num_workers

    def load_data(self):
        self.train_data = load_jsonl(self.train_path)
        self.valid_data = load_jsonl(self.valid_path)
        self.test_data = load_jsonl(self.test_path)

    def prepare_data(self):
        self.load_data()

    def create_dataset(self, data):
        dataset = NLUDataset(
            data, 
            tags2id=self.tags2id, 
            intents2id=self.intents2id,
            max_len=self.max_len
        )
        return dataset

    def create_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers, 
            collate_fn=NLUDataset.custom_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
            )

    def train_dataloader(self):
        train_dataset = self.create_dataset(data=self.train_data)
        return self.create_dataloader(train_dataset, shuffle=True)

    def val_dataloader(self):
        val_dataset = self.create_dataset(data=self.valid_data)
        return self.create_dataloader(val_dataset, shuffle=False)

    def test_dataloader(self):
        test_dataset = self.create_dataset(data=self.test_data)
        return self.create_dataloader(test_dataset, shuffle=False)

