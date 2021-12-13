from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from typing import Union, Dict

class DialogManager(object):
    def __init__(self):
        self.context_intents = {
            'PAST': ['value', 'aggregate'],
            'IF': ['account_change'],
            'OTHERS': ['help']
        }
        self.company = '005930'
        self._update_today()
        self.context = None
        self.intent = None
        self.knowledge = {
            'this year': +0,
            'last year': -1,
            'next year': +1
        }
        self.turns = defaultdict(dict)
        self.global_turn = 0
        self.turns[self.global_turn]['context'] = self.context
        self.turns[self.global_turn]['intent'] = self.intent


    def update_turn(self, turn):
        self.global_turn += 1

    def _update_today(self):
        self.today = dt.now()

    def _update_context(self, context:Union[str, None], intent:Union[str, None]):
        if (self.context is None) and (self.intent is None):
            self.context = context
            self.intent = intent
        elif (self.context != context) or (self.intent != intent):
            self.context = context if context is not None else self.context
            self.intent = intent if intent is not None else self.intent
            return True

        return False

    def post_process(self, nlu_results: dict) -> dict:
        """[summary]

        Args:
            nlu_results (dict): results from NLU module with 'intent' and 'tags'
                tags: {'DATE': ['last year']}

        Returns:
            dict: return the key information by tasks
        """
        self._update_today()
        
        context_intent = nlu_results['context_intent']
        tags = nlu_results['tags']

        if context_intent is not None:
            context, intent = context_intent.split('.')
        else: 
            context, intent = None, None
        change_in_context_intent = self._update_context(context, intent)
        return self._pots_process_key_information(context, intent, tags, change_in_context_intent)

    def _pots_process_key_information(self, context:Union[str, None], intent:Union[str, None], tags:dict, change_in_context_intent:bool):
        key_information = defaultdict()

        year = self._get_year(tags.get('date'))
        key_information['change_in_context_intent'] = change_in_context_intent
        key_information['context'] = context
        key_information['intent'] = intent
        key_information['year'] = year
        if context == 'PAST':
            key_information['account'] = tags.get('account')
        elif context == 'IF':
            key_information['subject_account']= tags.get('subject_account')
            key_information['target_account']= tags.get('target_account')
            key_information['subject_apply']= tags.get('subject_apply')
        else:
            key_information['context'] = context
            key_information['intent'] = intent
            # raise ValueError('Cannont find the context')
        return key_information

    def _get_year(self, date_keyword:Union[str, None]) -> int:
        if date_keyword is None:
            today_year = self.today.year
        else:
            try: 
                today_year = int(date_keyword)
            except ValueError:
                today_year = (self.today + relativedelta(years=self.knowledge[date_keyword])).year 
                
        if self.today >= dt.strptime(f'{today_year}-04-01', '%Y-%m-%d'):
            # true: year = 2020   2021.12.08 >= 2021.04.01 
            # false: year = 2019  2021.01.08 >= 2021.04.01 
            account_year = today_year - 1
        else:
            account_year = today_year - 2

        return account_year




