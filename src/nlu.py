from collections import defaultdict
from nlu_models import NLUModel
from nlu_utils import NLUTokenizer
import spacy
import json

class NLU(object):
    def __init__(self, checkpoint_path, labels_path):
        self.model = NLUModel.load_from_checkpoint(checkpoint_path)
        self.tokenizer = NLUTokenizer()
        with labels_path.open('r', encoding='utf-8') as file:
            ls = json.load(file)
        self.tags2id = ls['tags']
        self.intent2id = ls['intent']

    def __call__(self, text:str):
        nlu_results = defaultdict()
        text = text.lower()
        doc = self.sp_trf(text)
        tag = None
        words = None
        tags = set()
        for x in doc:
            # x.lemma_, x.ent_iob_, x.ent_type_
            if x.ent_iob_ == 'B':
                if tag is not None:
                    tags.add((tag.lower(), words))
                words = x.lemma_
                tag = x.ent_type_
            elif x.ent_iob_ == 'I':
                words += f' {x.lemma_}'
            else:
                if tag is not None:
                    tags.add((tag.lower(), words))
                tag = None
                words = None
        # assert only one tag for in the sentence

        nlu_results['tags'] = defaultdict()
        for tag, words in tags:
            # we assume only one tag per centence
            nlu_results['tags'][tag] = words

        # temproal: for testing
        nlu_results['doc'] = doc
        ## PAST
        if scenario == 1:
            nlu_results['context_intent'] = 'PAST.value'
            nlu_results['tags']['account'] = 'IS.CostOfSalesRatio'
        ## IF
        elif scenario == 2:
            nlu_results['context_intent'] = 'IF.account_change'
            nlu_results['tags']['subject_account'] = 'IS.CostOfSales'
            nlu_results['tags']['account'] = 'IS.OperatingIncome'
            nlu_results['tags']['subject_apply'] = ("increase", nlu_results['tags']['percent'])
        ## Embedded ML
        elif scenario == 3: 
            nlu_results['context_intent'] = 'EMB.forecast'
            nlu_results['tags']['account'] = 'IS.Revenue'
            nlu_results['tags']['model'] = 'linear'
        else:
            nlu_results['context_intent'] = None

        return nlu_results