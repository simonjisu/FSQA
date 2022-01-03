import json
import pickle
import pandas as pd
import numpy as np
from spacy.training.iob_utils import biluo_to_iob

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedShuffleSplit
from nlu_utils import NLUTokenizer
from rdflib import Graph, Literal, URIRef
from spacy.training import biluo_tags_to_offsets, iob_to_biluo
from datasets import load_dataset


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

def random_sampling(x_dict, x_key):
    idx_range = np.arange(len(x_dict[x_key]))
    idx = np.random.choice(idx_range, replace=False, p=np.ones(len(idx_range)) / len(idx_range))
    word, tag, desc = x_dict[x_key][idx]
    return word, tag, desc

def get_words_filtered(words, text):
    words_filtered = defaultdict(list)
    for k, v in words.items():
        for word, tag, desc in v:
            if desc != text:
                words_filtered[k].append((word, tag, desc))
    return words_filtered

def convert_to_string(x):
        if isinstance(x, URIRef):
            if len(x.split('#')) == 2:
                return x.split('#')[1]
            else:
                raise ValueError(f'Split error {x}')
        elif isinstance(x, Literal):
            return str(x)
        else:
            raise ValueError(f'Returned None')

def get_entity(s, x, tag):
    idx = s.index(x)
    return (idx, idx+len(x), tag)

def get_role_dict(sparql, knowledge):
    knowledge_query = sparql.get_predefined_knowledge(knowledge=knowledge)
    sparql_results = sparql.query(knowledge_query)
    role_dict = defaultdict(list)
    for s, p, o in sparql_results:
        s, p, o = map(convert_to_string, [s, p, o])
        if s == 'CalendarOneYear' or o == 'CalendarOneYear':
            continue
        if s not in role_dict[o]:
            role_dict[o].append(s)
        
    return role_dict

def process_successor(successors, role_dict, trg_acc, acc):
    if role_dict.get(acc) is None:
        # successors[trg_acc].extend(successors[acc])
        return None
    else:
        accs = role_dict.get(acc)
        if accs is not None:
            successors[trg_acc].extend(accs)
            for acc in accs:
                process_successor(successors, role_dict, trg_acc, acc)

def get_successor(onto, knowledge, exceptions=None):
    role_dict = get_role_dict(onto, knowledge=knowledge)
    successors = defaultdict(list)
    for trg_acc in role_dict.keys():
        if (exceptions is not None) and (trg_acc in exceptions):
            continue
        process_successor(successors, role_dict, trg_acc, trg_acc)
    return successors

def create_data():
    sparql = SparqlHandler(data_path / 'AccountRDF.xml')

    df_account = pd.read_csv(data_path / 'AccountName.csv', encoding='utf-8')
    ACC_DICT = defaultdict(dict)
    for _, row in df_account.iterrows():
        acc = row['acc']
        eng = row['acc_name_eng']
        kor = row['acc_name_kor']
        group = row['group']
        ACC_DICT[acc]['kor_name'] = kor
        ACC_DICT[acc]['eng_name'] = eng
        ACC_DICT[acc]['group'] = group

    # don't need to define the future but past words cannot use in future
    df = pd.read_csv(data_path / 'AccountWords.csv', encoding='utf-8')

    format_dict = {
        0: ['help'],
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

    words = defaultdict(list)
    for typ in ['year', 'quarter', 'words']:
        df_temp = df.loc[:, [typ, f'{typ}_tag', f'{typ}_desc']]
        df_temp = df_temp.loc[~df_temp[typ].isna(), :]
        for i, (w, t, desc) in df_temp.iterrows():
            words[typ].append((w, t, desc))

    exceptions = ['BalanceSheet', 'IncomeStatement', 'Ratios', 'CalendarOneYear']
    times = ['year', 'quarter']

    all_data = []
    
    # ----------------------------------------- target scenario: 1 ------------------------------------------------
    data1 = []
    trg_scenario = 1
    progress_bar = tqdm()
    words_filtered = get_words_filtered(words, text='FUTURE')
    for idx_fmt, fmt in enumerate(format_dict[trg_scenario]):
        
        for acc, dic in ACC_DICT.items():
            if acc in exceptions:
                continue
            target_account = dic['eng_name'].lower()
            knowledge, acc_type, _ = dic['group'].split('-')

            for t in ['year', 'quarter']:
                for t_word, t_tag, _ in words_filtered[t]:
                    entities = []
                    pre_token = np.random.choice(['what', 'how'], replace=False, p=np.ones(2)/2)
                    if idx_fmt == 0:
                        # what/how, target_account, [MASK] + year/quarter
                        # "{} is the {} in the {}?",
                        
                        s = fmt.format(
                            pre_token,
                            f_ENT(target_account), 
                            f_ENT(f'{t_word} {t}')
                            )
                    else:
                        # [MASK] + year/quarter, what/how, target_account
                        # "In the {}, {} is the value of the {}"
                        s = fmt.format(
                            f_ENT(f'{t_word} {t}'),
                            pre_token,
                            f_ENT(target_account)
                        )
                    # entities
                    ## target_account
                    # entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}.{acc_type}'))
                    entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}'))
                    ## MASK year/quarter
                    entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))
                    
                    data1.append(
                        {'text': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'PAST.value'} #, 'relation': relation}
                    )
                
                    progress_bar.update(1)
    
    # ----------------------------------------- target scenario: 2 ------------------------------------------------
    trg_scenario = 2
    bs_successors = get_successor(sparql, 'BS', exceptions)
    is_successors = get_successor(sparql, 'IS', exceptions)
    data2 = []
    n_sample = 7
    progress_bar = tqdm()
    words_filtered = get_words_filtered(words, text='FUTURE')

    for idx_fmt, fmt in enumerate(format_dict[trg_scenario]):
        for sub_tree in [bs_successors, is_successors]:
            for trg_acc, successors in sub_tree.items():
                if trg_acc in exceptions:
                    continue
                target_account = ACC_DICT[trg_acc]['eng_name'].lower()
                target_knowledge, target_acc_type, _ = ACC_DICT[trg_acc]['group'].split('-')
                for sub_acc in successors:
                    subject_account = ACC_DICT[sub_acc]['eng_name'].lower()
                    subject_knowledge, subject_acc_type, _ = ACC_DICT[trg_acc]['group'].split('-')
                    n = 0
                    while n < n_sample:
                        entities = []

                        apply_word, apply_tag, _ = random_sampling(x_dict=words_filtered, x_key='words')
                        t = np.random.choice(times, replace=False, p=np.ones(len(times))/len(times))
                        t_word, t_tag, _ = random_sampling(x_dict=words_filtered, x_key=t)
                        
                        number = np.random.randint(1, 99)
                        percent = np.random.choice(['percent', '%'], replace=False, p=np.ones(2)/2)
                        
                        if idx_fmt in [0, 1]:
                            # target_account, subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter
                            s = fmt.format(
                                f_ENT(target_account),
                                f_ENT(subject_account), 
                                f_ENT(apply_word), 
                                f_ENT(f'{number} {percent}'),
                                f_ENT(f'{t_word} {t}')
                                )
                            # relation = [1, 1, 2]
                        else:
                            # subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter, target_account
                            s = fmt.format(
                                f_ENT(subject_account), 
                                f_ENT(apply_word), 
                                f_ENT(f'{number} {percent}'),
                                f_ENT(f'{t_word} {t}'),
                                f_ENT(target_account)
                                )
                            # relation = [1, 2, 1]
                        # entities
                        ## target_account
                        # entities.append(get_entity(s, f_ENT(target_account), f'{target_knowledge}.{target_acc_type}'))
                        entities.append(get_entity(s, f_ENT(target_account), f'{target_knowledge}'))
                        ## subject_account
                        # entities.append(get_entity(s, f_ENT(subject_account), f'{subject_knowledge}.{subject_acc_type}'))
                        entities.append(get_entity(s, f_ENT(subject_account), f'{subject_knowledge}'))
                        ## MASK apply words
                        entities.append(get_entity(s, f_ENT(apply_word), apply_tag))
                        ## percentages
                        entities.append(get_entity(s, f_ENT(f'{number} {percent}'), 'PERCENT'))
                        ## MASK year/quarter
                        entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))

                        d = {'text': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'IF.fact'} #, 'relation': relation}
                        if d not in data2:
                            data2.append(
                                d
                            )
                        
                        progress_bar.update(1)
                        n += 1

    # ----------------------------------------- target scenario: 3 ------------------------------------------------
    data3 = []
    trg_scenario = 3
    progress_bar = tqdm()
    words_filtered = get_words_filtered(words, text='PAST')

    for fmt in format_dict[trg_scenario]:
        for acc, dic in ACC_DICT.items():
            if acc in exceptions:
                continue
            target_account = dic['eng_name'].lower()
            knowledge, acc_type, _ = dic['group'].split('-')
            for t in ['year', 'quarter']:
                for t_word, t_tag, _ in words_filtered[t]:
                    entities = []
                    s = fmt.format(
                        np.random.choice(['what', 'how']), 
                        f_ENT(target_account), 
                        f_ENT(f'{t_word} {t}')
                        )
                    # relation = [0, 0, 0]
                    # entities
                    ## target_account
                    # entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}.{acc_type}'))
                    entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}'))
                    ## MASK year/quarter
                    entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))
                    
                    data3.append(
                        {'text': s, 'entities': entities, 'intent': 'IF.forecast'} #, 'relation': relation}
                    )
                    
                    progress_bar.update(1)

    all_data = data1 + data2 + data3
    return all_data

def save_as_jsonl(data_list, path):
    with path.open('w', encoding='utf-8') as file:
        for line in tqdm(data_list, total=len(data_list), desc='saving'):
            file.write(json.dumps(line) + '\n')

def post_process(all_data):
    special_len = len(s_ENT)+len(e_ENT)

    for k, x in tqdm(enumerate(all_data), total=len(all_data)):
        all_data[k]['text'] = x['text'].replace(s_ENT, '').replace(e_ENT, '')
        for i, (s, e, ent) in enumerate(x['entities']):
            new_s = s-i*special_len
            new_e = new_s+(e-s)-special_len
            all_data[k]['entities'][i] = (new_s, new_e, ent)

    save_as_jsonl(all_data, path=data_path / 'all_data.jsonl')

def get_text_tags_intent(nlu_tokenizer, data):
    encodes = nlu_tokenizer.spacy_encode(data[0], pad_offset=False)
    tags = nlu_tokenizer.offsets_to_iob_tags(encodes['offset_mapping'], ents=data[1])
    return {'text': data[0], 'tags': tags, 'intent': data[2]}

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

    texts, entities, intents = list(zip(*all_data))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=1234)
    train_valid_idx, _ = list(*splitter.split(texts, intents))

    for idx in tqdm(range(len(texts)), total=len(texts), desc='spliting train&valid /test'):
        d = all_data[idx]
        if idx in train_valid_idx:
            # train_valid_data.append(all_data[idx])
            train_valid_data.append(d)
        else:
            # d = get_text_tags_intent(nlu_tokenizer, data=all_data[idx])
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
            # d = get_text_tags_intent(nlu_tokenizer, data=train_valid_data[idx])
            train_data.append(d)
        else:
            # d = get_text_tags_intent(nlu_tokenizer, data=train_valid_data[idx])
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
    
    s_ENT = '[E]'
    e_ENT = '[/E]'
    f_ENT = lambda x: f'{s_ENT}{x}{e_ENT}'

    main_path = Path('.').absolute().parent
    data_path = main_path / 'data'
    settings_path = main_path / 'setting_files'
    all_data = create_data()
    post_process(all_data)

    nlu_tokenizer = NLUTokenizer(hugg_path='bert-base-uncased', spacy_path='en_core_web_sm')
    
    ver = '1'
    # labels = {
    #     'intent': ['None', 'IF.fact', 'IF.forecast', 'PAST.value'],
    #     'tags': [
    #         'O', 'B-APPLY', 'I-APPLY', 
    #         'B-BS.Value', 'I-BS.Value', 'B-IS.Value', 'I-IS.Value', 
    #         'B-BS.Ratio', 'I-BS.Ratio', 'B-IS.Ratio', 'I-IS.Ratio',  
    #         'B-PERCENT', 'I-PERCENT', 'B-TIME', 'I-TIME'
    #     ] + ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    # }

    ver = '2'
    labels = {
        'intent': ['None', 'IF.fact', 'IF.forecast', 'PAST.value'],
        'tags': [
            'O', 'B-APPLY', 'I-APPLY', 'B-BS', 'I-BS', 'B-IS', 'I-IS',  
            'B-PERCENT', 'I-PERCENT', 'B-TIME', 'I-TIME'
        ] + ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    }

    process_all_data(nlu_tokenizer, ver=ver)
    with (data_path / f'labels{ver}.json').open('w', encoding='utf-8') as file:
        json.dump(labels, file)