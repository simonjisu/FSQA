from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from typing import Union
from pathlib import Path
import pandas as pd
from nlu_utils import NLUTokenizer
from sparql import SparqlHandler

class DialogManager(object):
    def __init__(self, words_path, acc_name_path, rdf_path):
        self.quarter_dict = {
            1: ('01.01', '03.31'), 2: ('04.01', '06.30'), 3: ('07.01', '09.30'), 4: ('10.01', '12.31')
        }
        self._update_today()

        self.tokenizer = NLUTokenizer()
        self._set_words_dict(words_path)
        self._set_acc_dict(acc_name_path)
        self._set_sparql(rdf_path)
        self.company = '005930'
        

        self.intent = None
        self.turns = defaultdict(dict)
        self.global_turn = 0
        # self.turns[self.global_turn]['intent'] = self.intent

        # Need to generalize this knowledge of time
        self.knowledge = {
            'this year': +0,
            'last year': -1,
            'next year': +1,
            'the 4th quarter': +0
        }

    def _set_words_dict(self, words_path):
        df = pd.read_csv(words_path, encoding='utf-8')
        self.words_dict = defaultdict(set)
        for typ in ['year', 'quarter', 'words']:
            df_temp = df.loc[:, [typ, f'{typ}_tag', f'{typ}_desc']]
            df_temp = df_temp.loc[~df_temp[typ].isna(), :]
            for _, (w, t, desc) in df_temp.iterrows():
                if typ in ['year', 'quarter']:
                    self.words_dict['TIME'].add((self.tokenizer.spacy_lemma(w), desc))
                else:
                    self.words_dict['APPLY'].add((self.tokenizer.spacy_lemma(w), desc))
        self.words_dict['TIME'].add(('this', 'PAST'))

    def _set_acc_dict(self, acc_name_path):
        df_account = pd.read_csv(acc_name_path, encoding='utf-8')
        self.is_account_names = []
        self.bs_account_names = []
        self.eng2acc = defaultdict()
        for _, row in df_account.iterrows():
            acc = row['acc']
            eng = row['acc_name_eng']
            knowledge = row['group'].split('-')[0]
            
            eng_lemma = self.tokenizer.spacy_lemma(eng.lower())
            if knowledge == 'IS':
                self.is_account_names.append(eng)
            elif knowledge == 'BS':
                self.bs_account_names.append(eng)
            self.eng2acc[eng_lemma] = acc

    def _set_sparql(self, rdf_path):
        self.sparql = SparqlHandler(rdf_path)
        self.bs_role = self.sparql.get_role_dict('BS')
        self.is_role = self.sparql.get_role_dict('IS')

    def _update_turn(self):
        self.turns[self.global_turn]['intent'] = self.intent
        self.global_turn += 1

    def _update_today(self):
        self.today = dt.strptime('2021.12.25', '%Y.%m.%d') # dt.now()

    def _update_intent(self, intent):
        self.intent = intent

    def post_process(self, nlu_results):
        """[summary]

        Args:
            nlu_results (dict): results from NLU module with 'intent' and 'tags'
                tags: {'DATE': ['last year']}

        Returns:
            dict: return the key information by tasks
        """
        self._update_today()
        intent = nlu_results['intent']
        tags = nlu_results['tags']
        print(tags)

        return self._pots_process_key_information(intent, tags)

    def _pots_process_key_information(self, intent, tags):
        error = False
        key_information = defaultdict()
        year, quarter = self._get_account_year(tags.get('TIME'))
        if year == True:
            error = year
            message = quarter
            return error, message
        
        key_information['intent'] = intent
        if intent == 'PAST.value':
            if len(tags['ACCOUNTS']) == 1:
                acc_name, knowlegde = tags.get('ACCOUNTS')[0]
                key_information['target_account'] = f'{knowlegde}.{self.eng2acc[acc_name]}'
                key_information['year'] = year
                key_information['quarter'] = '4Q' if quarter is None else f'{quarter}Q'  # if None sum all quarter values
            else:
                error = True
                return error, f'There are {len(tags["ACCOUNTS"])} tags in a sentence.'
                # raise ValueError('There are two tags in a sentence which one did you mean?')
        elif intent == 'IF.fact':
            # subject_account, target_account, apply
            if len(tags['ACCOUNTS']) > 1:
                key_information['year'] = year
                key_information['quarter'] = '4Q' if quarter is None else f'{quarter}Q'
                
                accs = tags.get('ACCOUNTS')
                a_acc_name, a_knowledge = accs[0]
                a_acc = self.eng2acc[a_acc_name]
                b_acc_name, b_knowledge = accs[1]
                b_acc = self.eng2acc[b_acc_name]
                is_trg_acc, is_sub_acc = self._get_target_subject_accounts(
                        self.is_role, a_acc, b_acc, a_knowledge, b_knowledge)
                bs_trg_acc, bs_sub_acc = self._get_target_subject_accounts(
                        self.bs_role, a_acc, b_acc, a_knowledge, b_knowledge)
                if (is_trg_acc is not None) and (is_sub_acc is not None):
                    key_information['target_account'] = is_trg_acc
                    key_information['subject_account'] = is_sub_acc
                if (bs_trg_acc is not None) and (bs_sub_acc is not None):
                    key_information['target_account'] = bs_trg_acc
                    key_information['subject_account'] = bs_sub_acc
                apply = tags.get('APPLY')
                percent = tags.get('PERCENT')

                mul, number = self._convert_apply_terms(apply, percent)
                if mul == True:
                    error = True
                    message = number
                    return error, message
                
                key_information['subject_apply'] = (mul, number)
            else:
                error = True
                return error, f'There are {len(tags["ACCOUNTS"])} tags in a sentence.'
        elif intent == 'IF.forecast':
            if len(tags['ACCOUNTS']) == 1:
                acc_name, knowlegde = tags.get('ACCOUNTS')[0]
                key_information['target_account'] = f'{knowlegde}.{self.eng2acc[acc_name]}'
                key_information['year'] = year
                key_information['quarter'] = '4Q' if quarter is None else f'{quarter}Q'
                key_information['model'] = 'linear'
            else:
                error = True
                return error, f'There are {len(tags["ACCOUNTS"])} tags in a sentence.'
        else:
            # None case
            key_information['target_account'] = None
            
        return error, key_information

    def _get_account_year(self, date_keyword:Union[str, None]) -> int:
        error = False
        recalculate_acc_year = False
        ref_Q = None
        if date_keyword is None:
            ref_year = self.today.year
        else:
            if 'year' in date_keyword:
                k = date_keyword.split(' ', 1)[0]
                desc = dict(self.words_dict['TIME']).get(k)
                if desc == 'PAST':
                    ref_year = self.today.year - 1
                    recalculate_acc_year = True
                elif desc == 'FUTURE':
                    ref_year = self.today.year + 1
                else:
                    ref_year = self.today.year
                    recalculate_acc_year = True
                ref_Q = None
            elif 'quarter' in date_keyword:
                k = date_keyword.split(' ', 1)[0]
                if k in ['tax', 'fiscal', 'financial', 'calendar']:
                    # raise error usually don't say these words
                    error = True
                    # return error, 'Need more specific time information'
                desc = dict(self.words_dict['TIME']).get(k)
                cur_Q = self._get_current_quarter()
                if desc == 'PAST':
                    recalculate_acc_year = True
                    if cur_Q == 1:
                        ref_year = self.today.year - 1
                        ref_Q = 4
                    else:
                        ref_year = self.today.year
                        ref_Q = cur_Q - 1
                elif desc == 'FUTURE':
                    if cur_Q == 4:
                        ref_year = self.today.year + 1
                        ref_Q = 1
                    else:
                        ref_year = self.today.year
                        ref_Q = cur_Q + 1
                else:
                    # REL case
                    for kq, ws in enumerate([['first', '1st'], ['second', '2nd'], ['third', '3rd'], ['fourth', '4th', 'final']], 1):
                        if k in ws:
                            break
                    # user's talking quarter = kq
                    ref_Q = kq
                    ref_year = self.today.year
                    recalculate_acc_year = True
            else:
                error = True

        if recalculate_acc_year:        
            if self.today >= dt.strptime(f'{self.today.year}.04.01', '%Y.%m.%d'):
                # still no report
                # true : acc_year = 2020  2021.12.08 >= 2021.04.01 
                # false: acc_year = 2019  2021.01.08 >= 2021.04.01 
                account_year = ref_year
            else:
                account_year = ref_year - 1 
        else:
            account_year = ref_year
        account_quarter = ref_Q

        if error:
            return error, 'No time error'
        return account_year, account_quarter

    def _get_current_quarter(self):
        for q, (s, e) in self.quarter_dict.items():
            s_time = dt.strptime(f'{self.today.year}.{s} 00:00:01', '%Y.%m.%d %H:%M:%S')
            e_time = dt.strptime(f'{self.today.year}.{e} 23:59:59', '%Y.%m.%d %H:%M:%S')
            if s_time <= self.today <= e_time:
                break
        return q

    def _get_target_subject_accounts(self, role_dict, a_acc, b_acc, a_knowledge, b_knowledge):
        def check(role_dict, acc, trg_acc):
            if trg_acc in role_dict[acc]:
                return True
            
            for sub_acc in role_dict[acc]:
                o = check(role_dict, sub_acc, trg_acc)
                if o:
                    return o
            else:
                return False

        target_account, subject_account = None, None
        if check(role_dict, a_acc, b_acc):
            target_account = f'{a_knowledge}.{a_acc}'
            subject_account = f'{b_knowledge}.{b_acc}'

        if check(role_dict, b_acc, a_acc):
            target_account = f'{b_knowledge}.{b_acc}'
            subject_account = f'{a_knowledge}.{a_acc}'

        return target_account, subject_account

    def _convert_apply_terms(self, apply, percent):
        if (apply is None) or (percent is None):
            return True, f'Missing information percent: {percent} / apply: {apply}'
        desc = dict(self.words_dict['APPLY']).get(apply)
        if desc == 'UP':
            sign = 1
        elif desc == 'DOWN':
            sign = -1
        else:
            return True, f'Not a proper word'

        number = 1 + sign*int(percent.split(' ')[0]) / 100

        return ('*', number)


