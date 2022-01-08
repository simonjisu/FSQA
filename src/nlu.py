from collections import defaultdict
from nlu_models import NLUModel
from nlu_utils import NLUTokenizer
import json

class NLUModule(object):
    def __init__(self, checkpoint_path, labels_path):
        self.model = NLUModel.load_from_checkpoint(checkpoint_path)
        self.tokenizer = NLUTokenizer()
        with labels_path.open('r', encoding='utf-8') as file:
            ls = json.load(file)
        self.id2tags = {v: k for k, v in ls['tags'].items()}
        self.id2intent = {v: k for k, v in ls['intent'].items()}

    def get_tags_intent(self, text):
        bert_encodes = self.tokenizer(
            text, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=64,
            return_tensors='pt'
        )
        o = self.model.predict(**bert_encodes)
        intent = list(map(self.id2intent.get, o['intent']))
        tags = list(map(self.id2tags.get, o['tags']))
        entities, doc = self.tokenizer.get_spacy_doc_entities(text, tags[1:-1])
        return entities, intent[0], doc

    def __call__(self, text:str):
        
        text = text.lower()
        ents, intent, doc = self.get_tags_intent(text)
        nlu_results = defaultdict()
        nlu_results['tags'] = defaultdict()
        nlu_results['intent'] = None if intent == 'None' else intent
        accounts = []
        for word, tag in ents:
            if tag in ['BS', 'IS']:
                accounts.append((word, tag))
            else:
                nlu_results['tags'][tag] = word

        nlu_results['tags']['ACCOUNTS'] = accounts
        nlu_results['doc'] = doc
        return nlu_results

        # if intent == 'PAST.value':
        #     if len(nlu_results['tags']) == 1:
        #         nlu_results['tags']['target_account'] = accounts[0]
        #     else:
        #         raise ValueError('There are two tags in a sentence which one did you mean?')
        # elif intent == 'IF.fact':
        #     # subject_account, target_account, apply
        #     nlu_results['tags']['accounts'] = accounts
        # elif intent == 'IF.forecast':
        #     if len(nlu_results['tags']) == 1:
        #         nlu_results['tags']['target_account'] = accounts[0]
        #     else:
        #         raise ValueError('There are two tags in a sentence which one did you mean?')
        # else:
        #     nlu_results
        
        
        # temproal: for testing
        # nlu_results['doc'] = doc

        ## PAST
        # if scenario == 1:
        #     nlu_results['context_intent'] = 'PAST.value'
        #     nlu_results['tags']['account'] = 'IS.CostOfSalesRatio'
        # ## IF
        # elif scenario == 2:
        #     nlu_results['context_intent'] = 'IF.account_change'
        #     nlu_results['tags']['subject_account'] = 'IS.CostOfSales'
        #     nlu_results['tags']['account'] = 'IS.OperatingIncome'
        #     nlu_results['tags']['subject_apply'] = ("increase", nlu_results['tags']['percent'])
        # ## Embedded ML
        # elif scenario == 3: 
        #     nlu_results['context_intent'] = 'EMB.forecast'
        #     nlu_results['tags']['account'] = 'IS.Revenue'
        #     nlu_results['tags']['model'] = 'linear'
        # else:
        #     nlu_results['context_intent'] = None

        return nlu_results