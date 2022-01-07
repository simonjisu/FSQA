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
from spacy.training import iob_to_biluo, biluo_to_iob, offsets_to_biluo_tags, biluo_tags_to_offsets
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



# class to NLUTokenizer
    # def str_to_offset_mapping(self, text: str) -> List[Tuple[int, int]]:   
    #     """tokenize and get offset mappings in the string

    #     Args:
    #         text (str): text are going to tokenized

    #     Returns:
    #         Dict[str, Union[List[str], List[Tuple[int, int]]]]: tokens, offset_mapping

    #         ```python
    #         spacy_encodes('How are you?')

    #         # tokens: ['How', 'are', 'you', '?']
    #         # offset_mapping: [(0, 3), (4, 7), (8, 11), (11, 12)]
    #         ```
    #     """
    #     tokens = []
    #     offset_mapping = []
    #     for x in self.spacy_nlp(text):
    #         token = str(x)
    #         tokens.append(token)
    #         offset_mapping.append((x.idx, x.idx+len(token)))
    #     return tokens, offset_mapping
    
    # @classmethod
    # def get_biluo_tags(cls, tokens, offset_mapping, ents):
    #     cur_start = 0
    #     state = "O" # Outside
    #     tags = []
    #     ent_idx = 0
    #     check_idx = 0
    #     for token in tokens:
    #         s_char, e_char, entity = ents[ent_idx]
    #         cur_offset_s, cur_offset_e = offset_mapping[check_idx]
    #         # check space
    #         if check_idx == 0:
    #             has_space = 1
    #         elif cur_start > cur_offset_s and cur_end < cur_offset_e:
    #             has_space = 0
    #         else:
    #             _, prev_offset_e = offset_mapping[check_idx-1]
    #             has_space = int(prev_offset_e != cur_offset_s)
    #         string_add = '_' if has_space else ''
            
    #         # deal with BERT's way of encoding spaces
    #         if token.startswith("##"):
    #             token = token[2:]
    #         else:
    #             token = string_add + token
    #         cur_end = cur_start + len(token)

    #         if state == "O" and cur_start <= s_char < cur_end:
    #             if cur_start == s_char and e_char == cur_end - has_space: # U- case
    #                 tags.append("U-" + entity)
    #                 ent_idx += 1
    #             else: # B- case
    #                 tags.append("B-" + entity)
    #                 state = "I-" + entity
    #             if cur_end > cur_offset_e:
    #                 check_idx += 1
    #         elif state.startswith("I-") and cur_start <= e_char:
    #             if cur_start > s_char and e_char == cur_end - has_space: # L- case
    #                 tags.append("L-" + entity)
    #                 check_idx += 1
    #                 state = "O"
    #                 ent_idx += 1
    #             else: # I- case
    #                 tags.append(state)
    #         else: # O- case
    #             tags.append(state)
    #             check_idx += 1
            
    #         cur_start = cur_end
    #         if ent_idx == len(ents):
    #             break

    #     tags += ['O'] * (len(tokens) - len(tags))
    #     return tags

    # @classmethod
    # def offsets_to_iob_tags(cls, offset_mapping, ents): #, get_acc_relation=False):
    #     """
    #     ```
    #     IOB SCHEME
    #     I - Token is inside an entity.
    #     O - Token is outside an entity.
    #     B - Token is the beginning of an entity.

    #     BILUO SCHEME
    #     B - Token is the beginning of a multi-token entity.
    #     I - Token is inside a multi-token entity.
    #     L - Token is the last token of a multi-token entity.
    #     U - Token is a single-token unit entity.
    #     O - Token is outside an entity.
    #     ```
    #     method: IOB SCHEME
    #     modified from https://github.com/explosion/spaCy/blob/9d63dfacfc85e7cd6db7190bd742dfe240205de5/spacy/training/iob_utils.py#L63

    #     encodes: batch encodes from huggingface TokenizerFast
    #     ents: entities with start & end characters in sentences + entity
    #     """
    #     # acc_relation = list()

    #     starts, ends = dict(), dict()
    #     for tkn_idx, (s_idx, e_idx) in enumerate(offset_mapping):
    #         if s_idx == e_idx == 0:
    #             continue
    #         starts[s_idx] = tkn_idx
    #         ends[e_idx] = tkn_idx
        
    #     char_in_ents = {}
    #     labels = ['-'] * len(offset_mapping)
    #     for s_char, e_char, ent in ents:
    #         if not ent:
    #             for s in starts:
    #                 labels[starts[s]] = 'O'
    #         else:
    #             for char_idx in range(s_char, e_char):
    #                 if char_idx in char_in_ents.keys():
    #                     raise ValueError(f'Trying to Overlapping same tokens: {char_in_ents[char_idx]} / {(s_char, e_char, ent)}')
    #                 char_in_ents[char_idx] = (s_char, e_char, ent)
    #             s_token = starts.get(s_char)
    #             e_token = ends.get(e_char)

    #             if s_token is not None and e_token is not None:
    #                 labels[s_token] = f'B-{ent}'
    #                 # add relation
    #                 # if get_acc_relation and len(ent.split('.')) > 1:
    #                 #     acc_relation.append((s_token, e_token+1))

    #                 for i in range(s_token + 1, e_token+1):
    #                     labels[i] = f'I-{ent}'
                        
    #     entity_chars = set()
    #     for s_char, e_char, ent in ents:
    #         for i in range(s_char, e_char):
    #             entity_chars.add(i)
    #     for token_idx, (s, e) in enumerate(offset_mapping):
    #         for i in range(s, e):
    #             if i in entity_chars:
    #                 break
    #         else:
    #             labels[token_idx] = 'O'
    #     if '-' in labels:
    #         raise ValueError('Some Tokens are not properly assigned' + f'{labels}')

    #     return labels

    # def pad_tags(self, input_ids, tags, pad_idx:int=0, is_pad_offset_before:bool=True):
    #     if is_pad_offset_before:
    #         padded_tags = tags + ([pad_idx] * (len(input_ids) - len(tags)))
    #     else:
    #         padded_tags = [pad_idx] * len(input_ids)
    #         j = 0
    #         for i, tkn_id in enumerate(input_ids):
    #             if tkn_id in self.bert.all_special_ids:
    #                 continue
    #             padded_tags[i] = tags[j]
    #             j += 1
    #             if j == len(tags):
    #                 break
    #     return padded_tags


    # def get_spanned_tags(self, 
    #     bert_offset_mapping:List[Tuple[int, int]], 
    #     spacy_offset_mapping:List[Tuple[int, int]],
    #     tags:List[str],
    #     # is_pad_offset_before:bool=True
    #     ):       
    #     """

    #     Args:
    #         bert_offset_mapping (List[Tuple[int, int]]): offset_mappings from self.bert
    #         spacy_offset_mapping (List[Tuple[int, int]]): offset_mapping from self.spacy_encode

    #     For example:
    #     ```
    #     question = 'when the trade and other current receivables drop by 53 percent in the financial year, 
    #     what will happen to the assets? [SEP] 23 .'
    #     spacy_offset_mapping = self.spacy_encodes(question)['offset_mapping']
    #     bert_offset_mapping = self.bert(question)['offset_mapping']
    #     ```
        
    #     ```
    #     Algorithm variables mapping
    #     bert_res |   i      |    j     |spanned_tags| c_e | c_s | check_idx
    #     [CLS]    | ( 0,  0) | ( 0,  4) | (  0,   0) |   0 |   0 | 1
    #     when     | ( 0,  4) | ( 0,  3) | (  0,   4) |   4 |   5 | 2
    #     the      | ( 0,  3) | ( 0,  5) | (  5,   8) |   8 |   9 | 3
    #     trade    | ( 0,  5) | ( 0,  3) | (  9,  14) |  14 |  15 | 4
    #     and      | ( 0,  3) | ( 0,  5) | ( 15,  18) |  18 |  19 | 5
    #     other    | ( 0,  5) | ( 0,  7) | ( 19,  24) |  24 |  25 | 6
    #     current  | ( 0,  7) | ( 0,  3) | ( 25,  32) |  32 |  33 | 7
    #     rec      | ( 0,  3) | ( 3,  5) | (  -,   -) |  36 |  33 | 7
    #     ##ei     | ( 3,  5) | ( 5, 10) | (  -,   -) |  38 |  33 | 7
    #     ##vable  | ( 5, 10) | (10, 11) | (  -,   -) |  43 |  33 | 7
    #     ##s      | (10, 11) | ( 0,  4) | ( 33,  44) |  44 |  45 | 8
    #     drop     | ( 0,  4) | ( 0,  2) | ( 45,  49) |  49 |  50 | 9
    #     by       | ( 0,  2) | ( 0,  2) | ( 50,  52) |  52 |  53 | 10
    #     53       | ( 0,  2) | ( 0,  7) | ( 53,  55) |  55 |  56 | 11
    #     percent  | ( 0,  7) | ( 0,  2) | ( 56,  63) |  63 |  64 | 12
    #     in       | ( 0,  2) | ( 0,  3) | ( 64,  66) |  66 |  67 | 13
    #     the      | ( 0,  3) | ( 0,  9) | ( 67,  70) |  70 |  71 | 14
    #     financial| ( 0,  9) | ( 0,  4) | ( 71,  80) |  80 |  81 | 15
    #     year     | ( 0,  4) | ( 0,  1) | ( 81,  85) |  85 |  85 | 16
    #     ,        | ( 0,  1) | ( 0,  4) | ( 85,  86) |  86 |  87 | 17
    #     what     | ( 0,  4) | ( 0,  4) | ( 87,  91) |  91 |  92 | 18
    #     will     | ( 0,  4) | ( 0,  6) | ( 92,  96) |  96 |  97 | 19
    #     happen   | ( 0,  6) | ( 0,  2) | ( 97, 103) | 103 | 104 | 20
    #     to       | ( 0,  2) | ( 0,  3) | (104, 106) | 106 | 107 | 21
    #     the      | ( 0,  3) | ( 0,  6) | (107, 110) | 110 | 111 | 22
    #     assets   | ( 0,  6) | ( 0,  1) | (111, 117) | 117 | 117 | 23
    #     ?        | ( 0,  1) | ( 0,  5) | (117, 118) | 118 | 119 | 24
    #     [SEP]    | ( 0,  5) | ( 0,  2) | (119, 124) | 124 | 125 | 25
    #     23       | ( 0,  2) | ( 0,  1) | (125, 127) | 127 | 128 | 26
    #     .        | ( 0,  1) | ( 0,  0) | (128, 129) | 129 | 130 | 27
    #     [SEP]    | ( 0,  0) | ( 0,  0) | (  0,   0) | 129 | 130 | 28
    #     ```
    #     """        
    #     spanned_tags = []
    #     i, j = 0, 1
    #     check_idx = 0
    #     current_s, current_e = 0, 0
    #     token_length = len(bert_offset_mapping)
    #     add_space = 1
    #     while i < token_length:
    #         s, e = bert_offset_mapping[i]
    #         if (j != token_length):
    #             s_next, e_next = bert_offset_mapping[j]
            
    #         # check have space
    #         _, e_check = spacy_offset_mapping[check_idx]
    #         if check_idx == len(spacy_offset_mapping)-1:
    #             s_check = e_check
    #         else:
    #             s_check, _ = spacy_offset_mapping[check_idx+1]
    #         add_space = 1 if e_check != s_check else 0
            
    #         if (s == 0 and e == 0) and (i == 0):
    #             # if bert tokenizer returns with add_special_tokens=True,
    #             # offset: start of sentence and end of sentence will always be s=0, e=0
    #             # so don't need to change current_s, current_e
    #             spanned_tags.append((0, 0))
    #             check_idx += 1
    #         # see next token is spanned or not
    #         elif (s == 0 and e != 0) and ((s_next == 0 and e_next != 0) or (s_next == 0 and e_next == 0)):
    #             # next one is not spanned, including the last token case
    #             current_e = current_s + e
    #             spanned_tags.append((current_s, current_e))
    #             current_s = current_e + add_space
    #             check_idx += 1
    #         elif (s == 0 and e != 0) and (s_next != 0 and e_next != 0):
    #             # next one is spanned, current is a spanned tag
    #             # stop change `current_s`, start to count how many spanned
    #             count = 1
    #             current_e = current_s + (e - s)
    #         elif (s != 0 and e != 0) and (s_next != 0 and e_next != 0):
    #             # in the middle of spanned tags
    #             current_e = current_e + (e - s)
    #             count += 1
    #         elif (s != 0 and e != 0) and ((s_next == 0 and e_next != 0) or (s_next == 0 and e_next == 0)):
    #             # next one is not spanned, current is the last spanned tag, including the last token case
    #             count += 1
    #             current_e = current_e + (e - s)
    #             spanned_tags += [(current_s, current_e)] * count
    #             current_s = current_e + add_space
    #             check_idx += 1
    #         else:
    #             # last token
    #             spanned_tags.append((0, 0))
    #             check_idx += 1

    #         i += 1
    #         j += 1
    #         if check_idx == len(spacy_offset_mapping):
    #             # break when checked all the tokens from spacy encodings
    #             break

    #     offset_dict = dict(zip(spacy_offset_mapping, tags))
    #     spanned_tags = list(map(offset_dict.get, spanned_tags))
    #     # if not is_pad_offset_before:
    #     #     return spanned_tags[1:-1]
    #     return spanned_tags
