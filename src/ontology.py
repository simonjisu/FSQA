
from rdflib import Graph, Literal, URIRef
from psycopg import sql
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
from collections import defaultdict
from embeddedML import LinearRegression
from utils import convert_to_string

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

    def get_role_dict(self, knowledge):
        knowledge_query = self.sparql.get_predefined_knowledge(knowledge=knowledge)
        sparql_results = self.sparql.query(knowledge_query)
        role_dict = defaultdict(list)
        for s, p, o in sparql_results:
            s, p, o = map(convert_to_string, [s, p, o])
            if s == 'CalendarOneYear' or o == 'CalendarOneYear':
                continue
            if s not in role_dict[o]:
                role_dict[o].append(s)
            
        return role_dict

    # TODO: move these to dialog manager or db handler
    def get_div_query(self, qs, acc):
        X, Y = qs
        if (X[0] == 'denominator') and (Y[0] == 'numerator'):
            B = X
            A = Y
        elif (Y[0] == 'denominator') and (X[0] == 'numerator'):
            B = Y
            A = X
        A_sign, A_node, A_q = A[1:]
        B_sign, B_node, B_q = B[1:]
        div_query_format = """
        SELECT ({} * CAST(A.{} AS REAL)) / ({} * CAST(B.{} AS REAL)) AS {}
        FROM (
            ( {} ) AS A
            JOIN 
            ( {} ) AS B ON 1=1
        ) 
        """
        div_query = sql.SQL(div_query_format).format(
            abs(A_sign), sql.Identifier(A_node.lower()), abs(B_sign), sql.Identifier(B_node.lower()), sql.Identifier(acc.lower()), A_q, B_q)
        return div_query

    def get_partof_query(self, qs, acc):
        # SELECT
        partof_query = sql.SQL("""SELECT """)
        partof_query += sql.SQL(' + ').join(
            [sql.SQL('({} * CAST({}.{} AS REAL)) ').format(sign, sql.Identifier(f'X{i}'), sql.Identifier(node.lower())) for i, (_, sign, node, _) in enumerate(qs)]
        )
        partof_query += sql.SQL(" AS {}").format(sql.Identifier(acc.lower()))
        # FROM
        partof_query += sql.SQL(" FROM ( ")
        for i, (*_, query) in enumerate(qs):
            partof_query += sql.SQL(f"( ") + query + sql.SQL(" ) AS {}").format(sql.Identifier(f'X{i}'))
            if i == 0:
                partof_query += sql.SQL(" JOIN ")
            elif i == (len(qs)-1):
                partof_query += sql.SQL(" ON 1=1 ")
            else:
                partof_query += sql.SQL(" ON 1=1 JOIN ")
        partof_query += sql.SQL(" )")
        return partof_query

class GraphDrawer():
    def __init__(
        self, 
        height='800px', 
        width='100%', 
        directed=True, 
        font_color='black', 
        heading='', 
        notebook=True,
        show_file_name='nx.html',
        label_name='eng_name'
    ):
        self.height = height
        self.width = width
        self.directed = directed
        self.font_color = font_color
        self.heading = heading
        self.notebook = notebook
        self.show_file_name = show_file_name
        self.label_name = label_name
        self.size_dict = {
            0: 20, 1: 18, 2: 16, 3: 14, 4: 12, 5: 10, 98: 12, 99: 10
        }

    def convert_to_string(self, x):
        if isinstance(x, URIRef):
            if len(x.split('#')) == 2:
                return x.split('#')[1]
            else:
                raise ValueError(f'Split error {x}')
        elif isinstance(x, Literal):
            return str(x)
        else:
            raise ValueError(f'Returned None')
    
    def get_graph(self, sparql_results, acc_dict, save=True):
        net = Network(
            height=self.height, 
            width=self.width, 
            directed=self.directed, 
            font_color=self.font_color, 
            heading=self.heading, 
            notebook=self.notebook
        )
        for src, link, trg in sparql_results:
            src = self.convert_to_string(src)
            link = self.convert_to_string(link)
            trg = self.convert_to_string(trg)

            src_label = acc_dict[src][self.label_name]
            trg_label = acc_dict[trg][self.label_name]
            src_group = acc_dict[src]['group']
            trg_group = acc_dict[trg]['group']
            src_fs, src_type, src_group = acc_dict[src]['group'].split('-')
            trg_fs, trg_type, trg_group = acc_dict[trg]['group'].split('-')
            src_title = f'Statement: {src_fs} Type: {src_type}'
            trg_title = f'Statement: {trg_fs} Type: {trg_type}'
            net.add_node(src, label=src_label, group=int(src_group), size=self.size_dict[int(src_group)], title=src_title)
            net.add_node(trg, label=trg_label, group=int(trg_group), size=self.size_dict[int(trg_group)], title=trg_title)
            net.add_edge(src, trg, weight=2, title=link)
        if save:
            net.show(self.show_file_name)
        else:
            return net

    def get_nx_graph(self, sparql_results, acc_dict):
        net = self.get_graph(sparql_results, acc_dict, False)
        nx_graph = nx.DiGraph(net.get_adj_list())
        nx_graph.add_edges_from([(x['from'], x['to'], {'label': x['title']}) for x in net.get_edges()])
        return nx_graph

class OntologySystem():
    def __init__(self, acc_name_path, rdf_path, model_path, kwargs_graph_drawer):
        
        self.graph_drawer = GraphDrawer(**kwargs_graph_drawer)
        self.sparql = SparqlHandler(rdf_path)
        self.embmodels = {
            'linear': LinearRegression(
                model_path, 
                scaler=lambda x: x / 1e13, 
                inv_scaler=lambda x: np.array(x*1e13).astype(np.int64)
            )
        }

        df_account = pd.read_csv(acc_name_path, encoding='utf-8')
        self.ACC_DICT = defaultdict(dict)
        for _, row in df_account.iterrows():
            acc = row['acc']
            eng = row['acc_name_eng']
            kor = row['acc_name_kor']
            group = row['group']
            self.ACC_DICT[acc]['kor_name'] = kor
            self.ACC_DICT[acc]['eng_name'] = eng
            self.ACC_DICT[acc]['group'] = group
        
        query_statement = """
        SELECT ?s ?p ?literal 
        WHERE { 
            ?s a acc:Account . 
            VALUES ?p { acc:Account_Property acc:Account_Level } 
            ?s ?p ?literal .
        }
        """
        qres = self.sparql.query(query_statement)
        for src, link, trg in qres:
            src = convert_to_string(src)
            link = convert_to_string(link)
            trg = convert_to_string(trg)
            self.ACC_DICT[src][link] = trg

    def get_graph(self, sparql_results, show=True):
        net = self.graph_drawer.get_graph(sparql_results, acc_dict=self.ACC_DICT, save=show)
        return net

    def get_nx_graph(self, sparql_results):
        return self.graph_drawer.get_nx_graph(sparql_results, acc_dict=self.ACC_DICT)

    def get_sub_tree_graph(self, sub_tree):
        sparql_results = self.sparql.get_sub_tree_relations(sub_tree)
        self.get_graph(sparql_results, show=True)

    def get_SQL(self, sparql_results, account, quarter, year, sub_account=None): 
        # sub_account: {acc_name: ('*', 1.1)}
        # TODO: ratio accounts should apply some functions
        sparql_results = list(map(lambda x: tuple(self.graph_drawer.convert_to_string(acc) for acc in x), list(sparql_results)))
        role_dict = defaultdict(list)
        for s, p, o in sparql_results:
            if s not in role_dict[o]:
                role_dict[o].append((s, p))
        
        # leaf node
        base_query_format = """SELECT (T.value){} AS {} FROM {} AS T """
        # search from top to bottom until reach all the leaf node
        query_dict = defaultdict(dict)
        for parent, children in role_dict.items():
            # start from account node
            for child, role in children:
                if child not in role_dict:
                    # leaf node
                    if query_dict[child].get('parents') is None:
                        query_dict[child]['parents'] = []

                    # check if has subject_account 
                    if  (sub_account is not None) and (child in sub_account):
                        apply, apply_number = sub_account[child]
                        sub_apply_query = sql.SQL(' ').join([sql.SQL(apply), apply_number])
                    else:
                        sub_apply_query = sql.SQL('')
                    
                    acc_knowledge = self.ACC_DICT[child]['group'].lower().split('-')[0]
                    view_table = f"vt_{acc_knowledge.lower()}_005930"
                    select_format = sql.SQL(base_query_format).format(sub_apply_query, sql.Identifier(child.lower()), sql.Identifier(view_table))
                    where_format = sql.SQL("""WHERE T.bsns_year = {} AND T.quarter = {} AND T.account = {}""").format(year, quarter, child)
                    query = select_format + where_format
                    
                    query_dict[child]['query'] = query
                    query_dict[child]['sign'] = 1.0 if self.ACC_DICT[child]['Account_Property'].lower() == 'positive' else -1.0
                    query_dict[child]['parents'].append((role, parent))

        # sort the nodes to make sure nodes with the leaf nodes comes first
        role_dict_sorted = dict(sorted(list(role_dict.items()), 
            key=lambda x: all([acc in list(query_dict.keys()) for acc, _ in x[1]]), reverse=True))
        
        for acc, childrens in role_dict_sorted.items():
            acc_sign = 1.0 if self.ACC_DICT[acc]['Account_Property'].lower() == 'positive' else -1.0
            query_dict[acc]['sign'] = acc_sign
            qs = []
            for child, role in childrens:
                qs.append(
                    (role,
                    query_dict[child]['sign'],
                    child,
                    query_dict[child]['query'])
                )
            if role.lower() != 'partof':
                query_dict[acc]['query'] = self.sparql.get_div_query(qs, acc)
            else:
                query_dict[acc]['query'] = self.sparql.get_partof_query(qs, acc)
            

        return query_dict[account]['query']