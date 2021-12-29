import json
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedShuffleSplit
from nlu_utils import NLUTokenizer
from ontology import GraphDrawer
from rdflib import Graph

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

def get_entity(s, x, tag):
    idx = s.index(x)
    return (idx, idx+len(x), tag)

def get_role_dict(sparql, knowledge):
    knowledge_query = sparql.get_predefined_knowledge(knowledge=knowledge)
    sparql_results = sparql.query(knowledge_query)
    role_dict = defaultdict(list)
    for s, p, o in sparql_results:
        s, p, o = map(GraphDrawer.convert_to_string, [s, p, o])
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
    trg_scenario = 1
    progress_bar = tqdm()
    for fmt in format_dict[trg_scenario]:
        for acc, dic in ACC_DICT.items():
            if acc in exceptions:
                continue
            target_account = dic['eng_name'].lower()
            knowledge, *_ = dic['group'].split('-')
            for t in times:
                for t_word, t_tag, t_desc in words[t]:
                    if t_desc != 'FUTURE':  # only add not future word in first scenario
                        entities = []

                        s = fmt.format(
                            np.random.choice(['what', 'how']),
                            f_ENT(target_account), 
                            f_ENT(f'{t_word} {t}')
                            )
                        relation = [0, 0, 0]  # no_relation, order1, order2
                        # entities
                        ## target_account
                        entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}.{acc}'))
                        ## MASK year/quarter
                        entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))
                        
                        all_data.append(
                            {'question': s, 'entities': entities, 'intent': 'PAST.value', 'relation': relation}
                        )
                        
                        progress_bar.update(1)

    
    # ----------------------------------------- target scenario: 2 ------------------------------------------------
    trg_scenario = 2
    bs_successors = get_successor(sparql, 'BS', exceptions)
    is_successors = get_successor(sparql, 'IS', exceptions)
    progress_bar = tqdm()
    for idx_fmt, fmt in enumerate(format_dict[trg_scenario]):
        for sub_tree in [bs_successors, is_successors]:
            for trg_acc, successors in sub_tree.items():
                if trg_acc in exceptions:
                    continue
                target_account = ACC_DICT[trg_acc]['eng_name'].lower()
                target_knowledge, *_ = ACC_DICT[trg_acc]['group'].split('-')
                for sub_acc in successors:
                    subject_account = ACC_DICT[sub_acc]['eng_name'].lower()
                    subject_knowledge, *_ = ACC_DICT[trg_acc]['group'].split('-')
                    for apply_word, apply_tag, apply_desc in words['words']:
                        for t in times:
                            for t_word, t_tag, t_desc in words[t]:
                                if t_desc != 'FUTURE':  # only add not future word in second scenario
                                    entities = []
                                    number = np.random.randint(1, 99)
                                    percent = np.random.choice(['percent', '%'])
                                    
                                    if idx_fmt in [0, 1]:
                                        # target_account, subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter
                                        s = fmt.format(
                                            f_ENT(target_account), 
                                            f_ENT(subject_account), 
                                            f_ENT(apply_word), 
                                            f_ENT(f'{number} {percent}'),
                                            f_ENT(f'{t_word} {t}')
                                            )
                                        relation = [1, 1, 2]
                                    else:
                                        # subject_account, [MASK], random_number + percent/%, [MASK] + year/quarter, target_account
                                        s = fmt.format(
                                            f_ENT(subject_account), 
                                            f_ENT(apply_word), 
                                            f_ENT(f'{number} {percent}'),
                                            f_ENT(f'{t_word} {t}'),
                                            f_ENT(target_account)
                                            )
                                        relation = [1, 2, 1]
                                    # entities
                                    ## target_account
                                    entities.append(get_entity(s, f_ENT(target_account), f'{target_knowledge}.{trg_acc}'))
                                    ## subject_account
                                    entities.append(get_entity(s, f_ENT(subject_account), f'{subject_knowledge}.{sub_acc}'))
                                    ## MASK apply words
                                    entities.append(get_entity(s, f_ENT(apply_word), apply_tag))
                                    ## percentages
                                    entities.append(get_entity(s, f_ENT(f'{number} {percent}'), 'PERCENT'))
                                    ## MASK year/quarter
                                    entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))
                        
                                    all_data.append(
                                        {'question': s, 'entities': sorted(entities, key=lambda x: x[0]), 'intent': 'IF.fact', 'relation': relation}
                                    )
                                    
                                    progress_bar.update(1)

    # ----------------------------------------- target scenario: 3 ------------------------------------------------
    trg_scenario = 3
    progress_bar = tqdm()
    for fmt in format_dict[trg_scenario]:
        for acc, dic in ACC_DICT.items():
            if acc in exceptions:
                continue
            target_account = dic['eng_name'].lower()
            knowledge, *_ = dic['group'].split('-')
            for t in times:
                for t_word, t_tag, t_desc in words[t]:
                    if t_desc != 'PAST':  # only add not past word in third scenario
                        entities = []
                        s = fmt.format(
                            np.random.choice(['what', 'how']), 
                            f_ENT(target_account), 
                            f_ENT(f'{t_word} {t}')
                            )
                        relation = [0, 0, 0]
                        # entities
                        ## target_account
                        entities.append(get_entity(s, f_ENT(target_account), f'{knowledge}.{acc}'))
                        ## MASK year/quarter
                        entities.append(get_entity(s, f_ENT(f'{t_word} {t}'), t_tag))
                        
                        all_data.append(
                            {'question': s, 'entities': entities, 'intent': 'IF.forecast', 'relation': relation}
                        )
                        
                        progress_bar.update(1)
    return all_data

def post_process(all_data):
    
    special_len = len(s_ENT)+len(e_ENT)

    for k, x in tqdm(enumerate(all_data), total=len(all_data)):
        all_data[k]['question'] = x['question'].replace(s_ENT, '').replace(e_ENT, '')
        for i, (s, e, ent) in enumerate(x['entities']):
            new_s = s-i*special_len
            new_e = new_s+(e-s)-special_len
            all_data[k]['entities'][i] = (new_s, new_e, ent)

    with (data_path / 'all_data.jsonl').open('w', encoding='utf-8') as file:
        for line in tqdm(all_data, total=len(all_data), desc='saving'):
            file.write(json.dumps(line) + '\n')
    
def process_all_data(nlu_tokenizer):
    

    with (data_path / 'all_data.jsonl').open('r', encoding='utf-8') as file:
        data = file.readlines()
        all_data = []
        for line in tqdm(data, total=len(data), desc='loading'):
            all_data.append(json.loads(line))

    processed_data = []
    for x in tqdm(all_data, total=len(all_data), desc='processing data'):
        encodes = nlu_tokenizer(text=x['question'], add_special_tokens=False, return_offsets_mapping=True)
        has_relation = x['relation'][0]
        tags, acc_relation = nlu_tokenizer.offsets_to_iob_tags(encodes, ents=x['entities'], get_acc_relation=has_relation)
        if acc_relation:
            # if there is relation process the coordinates of tokens
            # target: 1 / subject: 2
            # relation = [has_relation, target_coor, subject_coor]
            a, b = x['relation'][1:]
            if a == 1 and b == 2:
                s_trg, e_trg = acc_relation[0]
                s_sub, e_sub = acc_relation[1]
            elif a == 2 and b == 1:
                s_trg, e_trg = acc_relation[1]
                s_sub, e_sub = acc_relation[0]
            # plus 1 for add cls token in the front of sentences
            target_relation = (s_trg+1, e_trg+1)
            subject_relation = (s_sub+1, e_sub+1)
            relation = [has_relation, target_relation, subject_relation]
        else:
            relation = [has_relation, (0,0), (0,0)]
        
        processed_data.append((x['question'], tags, x['intent'], relation))

    with (data_path / 'all_data_processed.pickle').open('wb') as file:
        pickle.dump(processed_data, file)

def split_all_data():
    seed=777
    with (data_path / 'all_data_processed.pickle').open('rb') as file:
        all_data = pickle.load(file)

    questions, tags, intents, relations = list(zip(*all_data))
    # split to train & test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_idx, test_idx = list(*splitter.split(questions, intents))

    train_data = []
    test_data = []

    tags_set = set()
    for idx in tqdm(range(len(questions)), total=len(questions), desc='spliting data'):
        # process tags and intents
        data = (questions[idx], tags[idx], intents[idx], relations[idx])

        for t in tags[idx]:
            tags_set.add(t)

        if idx in train_idx:
            train_data.append(data)
        elif idx in test_idx:
            test_data.append(data)
        else:
            raise ValueError("Index Error")

    intents2id = {'None': 0}
    for intent in set(intents):
        if intents2id.get(intent) is None:
            intents2id[intent] = len(intents2id)

    tags2id = {'[PAD]': 0, 'O': 1}
    for t in tags_set:
        if tags2id.get(t) is None:
            tags2id[t] = len(tags2id)

    with (data_path / 'all_data_splitted.pickle').open('wb') as file:
        pickle.dump({
            'train': train_data, 
            'test': test_data, 
            }, file)

    with (data_path / 'all_data_ids.pickle').open('wb') as file:
        pickle.dump({
            'tags2id': tags2id, 
            'intents2id': intents2id
            }, file)


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
    process_all_data(nlu_tokenizer)
    split_all_data()