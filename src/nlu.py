from collections import defaultdict
import spacy

class NLU(object):
    def __init__(self, model_name='en_core_web_trf'):
        self.model_name = model_name
        self.sp_trf = spacy.load(model_name)

    def __call__(self, sentence:str, scenario:int):
        nlu_results = defaultdict()
        doc = self.sp_trf(sentence)
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
            nlu_results['tags']['account'] = 'IS.Revenue'
        elif scenario == 2:
            nlu_results['context_intent'] = 'PAST.value'
            nlu_results['tags']['account'] = 'IS.CostOfSalesRatio'
        ## IF
        elif scenario == 3: 
            nlu_results['context_intent'] = 'IF.account_change'
            nlu_results['tags']['subject_account'] = 'IS.CostOfSales'
            nlu_results['tags']['target_account'] = 'IS.OperatingIncome'
            nlu_results['tags']['subject_apply'] = lambda x: x*nlu_results['percent']
        else:
            nlu_results['context_intent'] = None
            

        return nlu_results