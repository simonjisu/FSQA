import json
import torch
import pandas as pd
import numpy as np
import os
import time
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForMaskedLM

from sklearn.model_selection import StratifiedShuffleSplit
from nlu_utils import NLUTokenizer
from rdflib import Graph, Literal, URIRef
from spacy.training import biluo_tags_to_offsets, iob_to_biluo, biluo_to_iob
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SparqlHandler():
    def __init__(self, rdf_path):
        self.graph = Graph()
        self.graph.load(rdf_path)

    def query(self, query_statement):
        return self.graph.query(query_statement)

    def get_related_nodes_from_sub_tree(self, sub_tree):
        nodes = set()
        for k, v in dict(sub_tree).items():
            for a in v:
                nodes.add(a)
            nodes.add(k)
        nodes = list(map(lambda x: f'acc:{x}', nodes))
        return nodes

    def get_sub_tree_relations(self, sub_tree):
        nodes = self.get_related_nodes_from_sub_tree(sub_tree)
        query_statement = """
        SELECT ?s ?p ?o 
        WHERE { 
            ?s rdf:type acc:Account .
            VALUES ?o { """ + f'{" ".join(nodes)}' + """ }
            VALUES ?p { acc:partOf acc:denominator acc:numerator } 
            ?s ?p ?o .
        }
        """
        return self.query(query_statement)

    def get_predefined_knowledge(self, knowledge:str):
        # BS, IS, BSR, ISR
        # TODO: 일부 노드 없음(balance sheet 에서 ratio의 분모인 sales )
        knowledge_queries = dict(
            BS="""
            SELECT ?s ?p ?o WHERE { 
            VALUES ?s { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShortTermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:EquitiesAbstract 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:PrepaidExpensesTurnoverPeriod acc:Ratios acc:CalendarOneYear acc:Revenue }
            VALUES ?o { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShortTermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:EquitiesAbstract 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:PrepaidExpensesTurnoverPeriod acc:Ratios acc:CalendarOneYear acc:Revenue }
            VALUES ?p { acc:partOf acc:denominator acc:numerator } 
            ?s ?p ?o .
            }
            """,
            IS="""
            SELECT ?s ?p ?o WHERE { 
            VALUES ?s { acc:BalanceSheet acc:Revenue acc:CostOfSales acc:GrossProfit acc:SellingGeneralAdministrativeExpenses 
            acc:OperatingIncome acc:FinanceIncome acc:FinancialExpenses acc:ProfitBeforeTax acc:IncomeTaxExpense acc:Profit 
            acc:IncomeStatement acc:CostOfSalesRatio acc:SellingGeneralAdministrativeRatio 
            acc:SalesAndSellingGeneralAdministrativeRatio acc:IncomeTaxRatio acc:ProfitRatio acc:Ratios }
            VALUES ?o { acc:BalanceSheet acc:Revenue acc:CostOfSales acc:GrossProfit acc:SellingGeneralAdministrativeExpenses 
            acc:OperatingIncome acc:FinanceIncome acc:FinancialExpenses acc:ProfitBeforeTax acc:IncomeTaxExpense acc:Profit 
            acc:IncomeStatement acc:CostOfSalesRatio acc:SellingGeneralAdministrativeRatio 
            acc:SalesAndSellingGeneralAdministrativeRatio acc:IncomeTaxRatio acc:ProfitRatio acc:Ratios }
            VALUES ?p { acc:partOf acc:denominator acc:numerator } 
            ?s ?p ?o .
            }
            """,
            ISR="""
            SELECT ?s ?p ?o WHERE { 
            VALUES ?s { acc:BalanceSheet acc:Revenue acc:CostOfSales acc:GrossProfit acc:SellingGeneralAdministrativeExpenses 
            acc:OperatingIncome acc:FinanceIncome acc:FinancialExpenses acc:ProfitBeforeTax acc:IncomeTaxExpense acc:Profit 
            acc:IncomeStatement acc:CostOfSalesRatio acc:SellingGeneralAdministrativeRatio 
            acc:SalesAndSellingGeneralAdministrativeRatio acc:IncomeTaxRatio acc:ProfitRatio acc:Ratios }
            VALUES ?o { acc:BalanceSheet acc:Revenue acc:CostOfSales acc:GrossProfit acc:SellingGeneralAdministrativeExpenses 
            acc:OperatingIncome acc:FinanceIncome acc:FinancialExpenses acc:ProfitBeforeTax acc:IncomeTaxExpense acc:Profit 
            acc:IncomeStatement acc:CostOfSalesRatio acc:SellingGeneralAdministrativeRatio 
            acc:SalesAndSellingGeneralAdministrativeRatio acc:IncomeTaxRatio acc:ProfitRatio acc:Ratios }
            VALUES ?p { acc:hasPart acc:isDenominatorOf acc:isNumeratorOf } 
            ?s ?p ?o .
            }
            """,
            BSR="""
            SELECT ?s ?p ?o WHERE { 
            VALUES ?s { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShortTermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:EquitiesAbstract 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:PrepaidExpensesTurnoverPeriod acc:Ratios acc:CalendarOneYear acc:Revenue }
            VALUES ?o { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShortTermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:EquitiesAbstract 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:PrepaidExpensesTurnoverPeriod acc:Ratios acc:CalendarOneYear acc:Revenue }
            VALUES ?p { acc:hasPart acc:isDenominatorOf acc:isNumeratorOf } 
            ?s ?p ?o .
            }
            """
        )
        return knowledge_queries[knowledge]

class DataCreator:
    s_ENT = '[E]'
    e_ENT = '[/E]'
    # f_ENT = lambda x: f'[E]{x}[/E]'

    def __init__(self, data_path, template_token_lengths=10, top_k=5, model_idx=3, simple_knowledge_tag=True):
        self.data_path = data_path
        self.exceptions = ['BalanceSheet', 'IncomeStatement', 'Ratios', 'CalendarOneYear']
        self.times = ['year', 'quarter']
        self.template_token_lengths = template_token_lengths
        self.top_k = top_k
        self.model_idx = model_idx
        self.simple_knowledge_tag = simple_knowledge_tag
        self.set_sparql()
        self.set_words_dict()
        self.set_format_dict()
        self.set_model_dict()
        self.f_ENT = lambda x: f'{self.s_ENT}{x}{self.e_ENT}'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.progress_bar = tqdm()

    def set_sparql(self):
        self.sparql = SparqlHandler(self.data_path / 'AccountRDF.xml')

        df_account = pd.read_csv(self.data_path / 'AccountName.csv', encoding='utf-8')
        self.ACC_DICT = defaultdict(dict)
        for _, row in df_account.iterrows():
            acc = row['acc']
            eng = row['acc_name_eng']
            kor = row['acc_name_kor']
            group = row['group']
            self.ACC_DICT[acc]['kor_name'] = kor
            self.ACC_DICT[acc]['eng_name'] = eng
            self.ACC_DICT[acc]['group'] = group

    def set_words_dict(self):
        df = pd.read_csv(self.data_path / 'AccountWords.csv', encoding='utf-8')
        self.words_dict = defaultdict(list)
        for typ in ['year', 'quarter', 'words']:
            df_temp = df.loc[:, [typ, f'{typ}_tag', f'{typ}_desc']]
            df_temp = df_temp.loc[~df_temp[typ].isna(), :]
            for _, (w, t, desc) in df_temp.iterrows():
                self.words_dict[typ].append((w, t, desc))

    def set_format_dict(self):
        self.format_dict = {
            0: [
                # only to train the ner task
                # account 
            ],
            1: [
                # what/how, target_account, [MASK] + year/quarter
                "{} is the {} in the {}?",
            ], 
            2: [
                # target_account, subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter
                "what happens to the {} when the {} {} by {} in the {}?",
                # target_account, subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter
                "what will be the effect to {} if the {} {} by {} in the {}?",
                # reverse the relation
                # subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter, target_account
                "when the {} {} by {} in the {}, what will happen to the {}?",
                # subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter, target_account
                "if the {} {} by {} in the {}, what will be the effect to {}?"
            ],
            3: [
                # what/how, target_account, [MASK] + year/quarter
                "{} will be the {} in the {}?"
            ]
        }

    def set_model_dict(self):
        self.model_dict = dict(enumerate([
            'bert-base-uncased',  # 0
            'albert-base-v2',  # 1
            'roberta-base',  # 2
            'google/electra-base-generator'  # 3
        ]))

    @classmethod
    def template_generator(cls, mask_token, max_length, acc_idxes, target_account):
        preset = [mask_token] * max_length
        candidates = []
        for i in acc_idxes:
            temp = deepcopy(preset)
            temp[i] = target_account
            candidates.append(' '.join(temp))
        return candidates

    def get_aug_sequences(self, tokenizer, model, cands, top_k, max_length):
        number_of_mask_token = max_length-1
        m_l = 0
        batch_size = 256
        n_workers = 0 if os.name == 'nt' else 4
        while m_l < number_of_mask_token:
            cands_tokens = [tokenizer.tokenize(c, truncation=True) for c in cands]
            infer_indices = []
            for c in cands_tokens:
                mask_idx = list(np.arange(len(c))[np.array(c) == '[MASK]'])
                choosed_idx = np.random.choice(mask_idx)
                infer_indices.append(choosed_idx)
            # batch output
            
            inputs = tokenizer(cands, padding=True, truncation=True, return_tensors='pt')
            ds = TensorDataset(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
            loader = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=n_workers)
            batch_cands = []
            
            self.progress_bar.reset(total=len(loader))
            self.progress_bar.set_postfix_str(s=f'{m_l}: {len(ds)}')
            for i, x in enumerate(loader):
                batch_infer_indices = infer_indices[(i*batch_size):((i+1)*batch_size)]
                input_ids = x[0].to(self.device)
                batch = {
                    'input_ids': input_ids,
                    'token_type_ids': x[1].to(self.device),
                    'attention_mask': x[2].to(self.device)
                }
                o = model(**batch)[0][:, 1:-1]
                _, sorted_idx = o.sort(dim=-1, descending=True)
                predict_ids = sorted_idx[torch.arange(sorted_idx.size(0)), batch_infer_indices, :top_k]
                cands_tensors = input_ids[:, 1:-1]
                cands_tensors = cands_tensors.unsqueeze(1).expand(cands_tensors.size(0), top_k, cands_tensors.size(1)).clone()
                cands_tensors[torch.arange(cands_tensors.size(0)), :, batch_infer_indices] = predict_ids

                
                batch_cands.extend([tokenizer.decode(c) for c in cands_tensors.view(cands_tensors.size(0)*top_k, -1).detach().cpu()])

                self.progress_bar.update(1)
            self.progress_bar.refresh()

            cands = batch_cands
            m_l += 1

        aug_cands = []
        for c in cands:
            for tkns in tokenizer(c, add_special_tokens=False)['input_ids']:
                aug_cand = tokenizer.decode(tkns, skip_special_tokens=True)
                aug_cands.append(aug_cand)
        #     for special_tkn in tokenizer.all_special_tokens:
        #         c = c.replace(special_tkn, '').strip()
        #     aug_cands.append(c)
        # cands = [c.replace(tokenizer.pad_token, '').replace(tokenizer.sep_token, '').strip() for c in cands]
        return aug_cands

    @classmethod
    def get_words_filtered(cls, words, text):
        words_filtered = defaultdict(list)
        for k, v in words.items():
            for word, tag, desc in v:
                if desc != text:
                    words_filtered[k].append((word, tag, desc))
        return words_filtered
    
    @classmethod
    def convert_to_string(cls, x):
        if isinstance(x, URIRef):
            if len(x.split('#')) == 2:
                return x.split('#')[1]
            else:
                raise ValueError(f'Split error {x}')
        elif isinstance(x, Literal):
            return str(x)
        else:
            raise ValueError(f'Returned None')

    @classmethod
    def random_sampling(cls, x_dict, x_key):
        idx_range = np.arange(len(x_dict[x_key]))
        idx = np.random.choice(idx_range, replace=False, p=np.ones(len(idx_range)) / len(idx_range))
        word, tag, desc = x_dict[x_key][idx]
        return word, tag, desc

    def get_role_dict(self, knowledge):
        knowledge_query = self.sparql.get_predefined_knowledge(knowledge=knowledge)
        sparql_results = self.sparql.query(knowledge_query)
        role_dict = defaultdict(list)
        for s, p, o in sparql_results:
            s, p, o = map(self.convert_to_string, [s, p, o])
            if s == 'CalendarOneYear' or o == 'CalendarOneYear':
                continue
            if s not in role_dict[o]:
                role_dict[o].append(s)
            
        return role_dict

    def process_successor(self, successors, role_dict, trg_acc, acc):
        if role_dict.get(acc) is None:
            # successors[trg_acc].extend(successors[acc])
            return None
        else:
            accs = role_dict.get(acc)
            if accs is not None:
                successors[trg_acc].extend(accs)
                for acc in accs:
                    self.process_successor(successors, role_dict, trg_acc, acc)

    def get_successor(self, knowledge, exceptions=None):
        role_dict = self.get_role_dict(knowledge=knowledge)
        successors = defaultdict(list)
        for trg_acc in role_dict.keys():
            if (exceptions is not None) and (trg_acc in exceptions):
                continue
            self.process_successor(successors, role_dict, trg_acc, trg_acc)
        return successors

    def scenario_0(self):
        data = []
        trg_scenario = 0
        
        count = 0

        model_path = self.model_dict[self.model_idx]
        model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        mask_token = tokenizer.mask_token
        self.progress_bar.set_description_str(desc=f'Scenario {trg_scenario}')
        for acc, dic in self.ACC_DICT.items():
            if acc in self.exceptions:
                continue
            target_account = dic['eng_name'].lower()
            knowledge, acc_type, _ = dic['group'].split('-')
            target_entitiy = f'{knowledge}' if self.simple_knowledge_tag else f'{knowledge}.{acc_type}'
            for l in range(1, self.template_token_lengths+1):
                if l == 1:
                    # skip fill mask
                    sentences = [self.f_ENT(target_account)]
                else:
                    # fill mask
                    acc_idxes = np.random.choice(range(l), size=(l,), replace=False)

                    cands = self.template_generator(
                        mask_token=mask_token, 
                        max_length=l,
                        acc_idxes=acc_idxes,
                        target_account=target_account
                    )
                    sentences = self.get_aug_sequences(
                        tokenizer=tokenizer, 
                        model=model, 
                        cands=cands, 
                        top_k=self.top_k, 
                        max_length=l
                    )
                    sentences = [
                        f'{self.f_ENT(target_account)}'.join(aug_sent.split(target_account)) for aug_sent in sentences
                    ]
                for s in sentences:
                    entities = []
                    entities.append(get_entity(s, self.f_ENT(target_account), target_entitiy))
                    data.append(
                        {'text': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'None'} 
                    )
                    count += 1
                    self.progress_bar.set_description_str(desc=f'Scenario {trg_scenario} acc={target_account} len={l} # {count}')
        self.progress_bar.refresh()
        return data

    def scenario_1(self):
        data = []
        trg_scenario = 1
        words_filtered = self.get_words_filtered(self.words_dict, text='FUTURE')
        self.progress_bar.reset(0)
        self.progress_bar.set_description_str(desc=f'Scenario {trg_scenario}')
        for idx_fmt, fmt in enumerate(self.format_dict[trg_scenario]):
            for acc, dic in self.ACC_DICT.items():
                if acc in self.exceptions:
                    continue
                target_account = dic['eng_name'].lower()
                knowledge, acc_type, _ = dic['group'].split('-')
                target_entitiy = f'{knowledge}' if self.simple_knowledge_tag else f'{knowledge}.{acc_type}'
                self.progress_bar.set_postfix_str(s=f'{target_account}')
                for t in self.times:
                    for t_word, t_tag, _ in words_filtered[t]:
                        
                        entities = []
                        pre_token = np.random.choice(['what', 'how'], replace=False, p=np.ones(2)/2)
                        if idx_fmt == 0:
                            # what/how, target_account, [MASK] + year/quarter
                            # "{} is the {} in the {}?",
                            s = fmt.format(
                                pre_token,
                                self.f_ENT(target_account), 
                                self.f_ENT(f'{t_word} {t}')
                            )
                        else:
                            # [MASK] + year/quarter, what/how, target_account
                            # "In the {}, {} is the value of the {}"
                            s = fmt.format(
                                self.f_ENT(f'{t_word} {t}'),
                                pre_token,
                                self.f_ENT(target_account)
                            )
                        # entities
                        ## target_account
                        entities.append(get_entity(s, self.f_ENT(target_account), target_entitiy))
                        ## MASK year/quarter
                        entities.append(get_entity(s, self.f_ENT(f'{t_word} {t}'), t_tag))
                        
                        data.append(
                            {'text': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'PAST.value'} 
                        )
                        self.progress_bar.update(1)
        self.progress_bar.refresh()         
        return data

    def scenario_2(self):
        n_sample = 10
        data = []
        trg_scenario = 2
        self.progress_bar.reset(0)
        self.progress_bar.set_description_str(desc=f'Scenario {trg_scenario}')
        bs_successors = self.get_successor('BS', self.exceptions)
        is_successors = self.get_successor('IS', self.exceptions)
        words_filtered = self.get_words_filtered(self.words_dict, text='FUTURE')

        for idx_fmt, fmt in enumerate(self.format_dict[trg_scenario]):
            for sub_tree in [bs_successors, is_successors]:
                for trg_acc, successors in sub_tree.items():
                    if trg_acc in self.exceptions:
                        continue
                    target_account = self.ACC_DICT[trg_acc]['eng_name'].lower()
                    target_knowledge, target_acc_type, _ = self.ACC_DICT[trg_acc]['group'].split('-')
                    target_entitiy = f'{target_knowledge}' if self.simple_knowledge_tag else f'{target_knowledge}.{target_acc_type}'

                    for sub_acc in successors:
                        subject_account = self.ACC_DICT[sub_acc]['eng_name'].lower()
                        subject_knowledge, subject_acc_type, _ = self.ACC_DICT[trg_acc]['group'].split('-')
                        subject_entitiy = f'{subject_knowledge}' if self.simple_knowledge_tag else f'{subject_knowledge}.{subject_acc_type}'
                        self.progress_bar.set_postfix_str(s=f'{target_account} / {subject_account}')
                        n = 0
                        while n < n_sample:

                            entities = []

                            apply_word, apply_tag, _ = self.random_sampling(x_dict=words_filtered, x_key='words')
                            t = np.random.choice(self.times, replace=False, p=np.ones(len(self.times))/len(self.times))
                            t_word, t_tag, _ = self.random_sampling(x_dict=words_filtered, x_key=t)
                            
                            number = np.random.randint(1, 99)
                            percent = np.random.choice(['percent', '%'], replace=False, p=np.ones(2)/2)
                            
                            if idx_fmt in [0, 1]:
                                # target_account, subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter
                                s = fmt.format(
                                    self.f_ENT(target_account),
                                    self.f_ENT(subject_account), 
                                    self.f_ENT(apply_word), 
                                    self.f_ENT(f'{number} {percent}'),
                                    self.f_ENT(f'{t_word} {t}')
                                    )
                                # relation = [1, 1, 2]
                            else:
                                # subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter, target_account
                                s = fmt.format(
                                    self.f_ENT(subject_account), 
                                    self.f_ENT(apply_word), 
                                    self.f_ENT(f'{number} {percent}'),
                                    self.f_ENT(f'{t_word} {t}'),
                                    self.f_ENT(target_account)
                                    )
                                # relation = [1, 2, 1]
                            # entities
                            ## target_account
                            entities.append(get_entity(s, self.f_ENT(target_account), target_entitiy))
                            ## subject_account
                            entities.append(get_entity(s, self.f_ENT(subject_account), subject_entitiy))
                            ## MASK apply words
                            entities.append(get_entity(s, self.f_ENT(apply_word), apply_tag))
                            ## percentages
                            entities.append(get_entity(s, self.f_ENT(f'{number} {percent}'), 'PERCENT'))
                            ## MASK year/quarter
                            entities.append(get_entity(s, self.f_ENT(f'{t_word} {t}'), t_tag))

                            d = {'text': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'IF.fact'} #, 'relation': relation}
                            if d not in data:
                                data.append(d)
                            
                            self.progress_bar.update(1)
                            n += 1
        self.progress_bar.refresh()
        return data
    
    def scenario_3(self):
        data = []
        trg_scenario = 3
        self.progress_bar.reset(0)
        self.progress_bar.set_description_str(desc=f'Scenario {trg_scenario}')
        words_filtered = self.get_words_filtered(self.words_dict, text='PAST')

        for fmt in self.format_dict[trg_scenario]:
            for acc, dic in self.ACC_DICT.items():
                if acc in self.exceptions:
                    continue
                target_account = dic['eng_name'].lower()
                knowledge, acc_type, _ = dic['group'].split('-')
                target_entitiy = f'{knowledge}' if self.simple_knowledge_tag else f'{knowledge}.{acc_type}'
                self.progress_bar.set_postfix_str(s=f'{target_account}')
                for t in self.times:
                    for t_word, t_tag, _ in words_filtered[t]:

                        entities = []
                        s = fmt.format(
                            np.random.choice(['what', 'how']), 
                            self.f_ENT(target_account), 
                            self.f_ENT(f'{t_word} {t}')
                            )
                        # entities
                        ## target_account
                        entities.append(get_entity(s, self.f_ENT(target_account), target_entitiy))
                        ## MASK year/quarter
                        entities.append(get_entity(s, self.f_ENT(f'{t_word} {t}'), t_tag))
                        
                        data.append(
                            {'text': s, 'entities': entities, 'intent': 'IF.forecast'} #, 'relation': relation}
                        )
                        self.progress_bar.update(1)
        self.progress_bar.refresh()
        return data

    def post_process(self, all_data):
        special_len = len(self.s_ENT)+len(self.e_ENT)
        for k, x in tqdm(enumerate(all_data), total=len(all_data), desc='Post Process'):
            all_data[k]['text'] = x['text'].replace(self.s_ENT, '').replace(self.e_ENT, '')
            for i, (s, e, ent) in enumerate(x['entities']):
                new_s = s-i*special_len
                new_e = new_s+(e-s)-special_len
                all_data[k]['entities'][i] = (new_s, new_e, ent)

    def create_data(self):
        data0 = self.scenario_0()
        data1 = self.scenario_1()
        data2 = self.scenario_2()
        data3 = self.scenario_3()
        all_data = data0 + data1 + data2 + data3
        print(f'total created: {len(all_data)}')
        self.post_process(all_data)
        save_as_jsonl(all_data, path=self.data_path / 'all_data.jsonl')

def get_entity(s, x, tag):
    idx = s.index(x)
    return (idx, idx+len(x), tag)

def save_as_jsonl(data_list, path):
    with path.open('w', encoding='utf-8') as file:
        for line in tqdm(data_list, total=len(data_list), desc='saving'):
            file.write(json.dumps(line) + '\n')

def process_all_data(nlu_tokenizer, ver=''):
    
    with (data_path / 'all_data.jsonl').open('r', encoding='utf-8') as file:
        data = file.readlines()
        all_data = []
        for line in tqdm(data, total=len(data), desc='loading'):
            all_data.append(json.loads(line))

    # custom data
    # split to train/valid & test
    train_valid_data = []
    test_data = []

    texts, _, intents = list(zip(*all_data))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=1234)
    train_valid_idx, _ = list(*splitter.split(texts, intents))

    for idx in tqdm(range(len(texts)), total=len(texts), desc='spliting train&valid /test'):
        d = all_data[idx]
        if idx in train_valid_idx:
            train_valid_data.append(d)
        else:
            test_data.append(d)

    # split to train valid
    train_data = []
    valid_data = []

    tv_texts, _, tv_intents = list(zip(*train_valid_data))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
    train_idx, _ = list(*splitter.split(tv_texts, tv_intents))

    for idx in tqdm(range(len(tv_texts)), total=len(tv_texts), desc='spliting train/valid'):
        d = train_valid_data[idx]
        if idx in train_idx:
            train_data.append(d)
        else:
            valid_data.append(d)

    # conll2003
    conll = load_dataset('conll2003')
    def add_conll_data(conll, nlu_tokenizer, data_list, typ='train'):
        dataset = conll[typ]
        feature = dataset.features['ner_tags'].feature
        errors = 0
        for x in tqdm(dataset, total=len(dataset), desc=f'{typ}set'):
            text = ' '.join(x['tokens']).lower()
            doc = nlu_tokenizer.spacy_nlp(text)

            tags = list(map(feature.int2str, x['ner_tags']))
            spacy_tokens = list(map(str, doc))
            original_tokens = list(map(str.lower, x['tokens']))
            mapped_tags = nlu_tokenizer.fix_tags_alignment(
                longer_tokens=spacy_tokens, shorter_tokens=original_tokens, tags=tags
            )

            entities = biluo_tags_to_offsets(doc, mapped_tags)
            if not entities:
                errors += 1
                continue

            d = {'text': text, 'entities': entities, 'intent': 'None'}
            data_list.append(d)
        print(f'{typ} errors: {errors} / {len(dataset)}')

    add_conll_data(conll, nlu_tokenizer, train_data, typ='train')
    add_conll_data(conll, nlu_tokenizer, valid_data, typ='validation')
    add_conll_data(conll, nlu_tokenizer, test_data, typ='test')
    
    save_as_jsonl(train_data, path=data_path / f'all_data_train{ver}.jsonl')
    save_as_jsonl(valid_data, path=data_path / f'all_data_valid{ver}.jsonl')
    save_as_jsonl(test_data, path=data_path / f'all_data_test{ver}.jsonl')

if __name__ == '__main__':
    import argparse
    # s_ENT = '[E]'
    # e_ENT = '[/E]'
    # f_ENT = lambda x: f'{s_ENT}{x}{e_ENT}'

    main_path = Path('.').absolute().parent
    data_path = main_path / 'data'
    settings_path = main_path / 'setting_files'

    labels_version = {
        '_complex': {
            'intent': ['None', 'IF.fact', 'IF.forecast', 'PAST.value'],
            'tags': [
                'O', 'B-APPLY', 'I-APPLY', 
                'B-BS.Value', 'I-BS.Value', 'B-IS.Value', 'I-IS.Value', 
                'B-BS.Ratio', 'I-BS.Ratio', 'B-IS.Ratio', 'I-IS.Ratio',  
                'B-PERCENT', 'I-PERCENT', 'B-TIME', 'I-TIME'
            ] + ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        },
        '_simple': {
            'intent': ['None', 'IF.fact', 'IF.forecast', 'PAST.value'],
            'tags': [
                'O', 'B-APPLY', 'I-APPLY', 'B-BS', 'I-BS', 'B-IS', 'I-IS',  
                'B-PERCENT', 'I-PERCENT', 'B-TIME', 'I-TIME'
            ] + ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        }
    }
    parser = argparse.ArgumentParser(description='settings data creation')
    parser.add_argument('-l', '--template_token_lengths', type=int, default=5,
                        help='template_token_lengths')
    parser.add_argument('-tk', '--top_k', type=int, default=5,
                        help='top_k')
    parser.add_argument('-mi', '--model_idx', type=int, default=3,
                        help='model_idx')
    parser.add_argument('-s', '--simple_knowledge_tag', action='store_true',
                        help='simple_knowledge_tag')
    args = parser.parse_args()
    
    template_token_lengths=args.template_token_lengths
    top_k=args.top_k
    model_idx=args.model_idx
    simple_knowledge_tag=args.simple_knowledge_tag

    ver = '_simple' if simple_knowledge_tag else '_complex'
    start = time.time()
    creator = DataCreator(
        data_path, 
        template_token_lengths=template_token_lengths, 
        top_k=top_k, 
        model_idx=model_idx, 
        simple_knowledge_tag=simple_knowledge_tag
    )
    creator.create_data()

    nlu_tokenizer = NLUTokenizer(hugg_path='bert-base-uncased', spacy_path='en_core_web_sm')

    process_all_data(nlu_tokenizer, ver=ver)
    with (data_path / f'labels{ver}.json').open('w', encoding='utf-8') as file:
        json.dump(labels_version[ver], file)
    end = time.time()

    print(f'total: {(end-start) / 60} min')