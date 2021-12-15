
from rdflib import Graph, Literal, URIRef
import networkx as nx
import pandas as pd
from pyvis.network import Network
from collections import defaultdict

class SparqlHandler():
    def __init__(self, rdf_path):
        self.graph = Graph()
        self.graph.load(rdf_path)

    def query(self, query_statement):
        return self.graph.query(query_statement)

    def get_sub_tree_relations(self, sub_tree):
        nodes = set()
        for k, v in dict(sub_tree).items():
            for a in v:
                nodes.add(a)
            nodes.add(k)
        nodes = list(map(lambda x: f'acc:{x}', nodes))
        query_statement = """
        SELECT ?s ?p ?o 
        WHERE { 
            ?s rdf:type acc:Account .
            VALUES ?o { """ + f'{" ".join(nodes)}' + """ }
            VALUES ?p { acc:partOf acc:denominator acc:numerator } 
            ?s ?p ?o .
        }
        """
        return self.graph.query(query_statement)

    def get_predefined_knowledge(self, knowledge:str):
        # BS, IS, BSR, ISR
        knowledge_queries = dict(
            BS="""
            SELECT ?s ?p ?o WHERE { 
            VALUES ?s { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShorttermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:TotalEquity 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:Ratios acc:CalendarOneYear }
            VALUES ?o { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShorttermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:TotalEquity 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:Ratios acc:CalendarOneYear }
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
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShorttermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:TotalEquity 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:Ratios acc:CalendarOneYear }
            VALUES ?o { acc:CurrentAssets acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses 
            acc:Inventories acc:NoncurrentAssets acc:PropertyPlantAndEquipment acc:IntangibleAssets acc:AssetsAbstract 
            acc:CurrentLiabilities acc:TradeAndOtherCurrentPayables acc:ShorttermBorrowings acc:AdvancesCustomers 
            acc:NoncurrentLiabilities acc:BondsIssued acc:LongTermBorrowings acc:LiabilitiesAbstract acc:TotalEquity 
            acc:LiabilitiesAndEquities acc:BalanceSheet acc:IncomeStatement acc:TradeReceivableTurnoverPeriod 
            acc:InventoriesTurnoverPeriod acc:TradePayablesTurnoverPeriod acc:AdvancesCustomersTurnoverPeriod 
            acc:Ratios acc:CalendarOneYear }
            VALUES ?p { acc:hasPart acc:isDenominatorOf acc:isNumeratorOf } 
            ?s ?p ?o .
            }
            """
        )
        return knowledge_queries[knowledge]

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
    ):
        self.height = height
        self.width = width
        self.directed = directed
        self.font_color = font_color
        self.heading = heading
        self.notebook = notebook
        self.show_file_name = show_file_name
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

            src_label = acc_dict[src]['name']
            trg_label = acc_dict[trg]['name']
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
    def __init__(self, acc_name_path,rdf_path,kwargs_graph_drawer):
        
        self.graph_drawer = GraphDrawer(**kwargs_graph_drawer)
        self.sparql = SparqlHandler(rdf_path)

        df_account = pd.read_csv(acc_name_path, encoding='utf-8')
        self.ACC_DICT = defaultdict(dict)
        for _, row in df_account.iterrows():
            eng = row['acc_name_eng']
            kor = row['acc_name_kor']
            group = row['group']
            self.ACC_DICT[eng]['name'] = kor
            self.ACC_DICT[eng]['group'] = group
        self.ACC_DICT['CalendarOneYear']['name'] = '365 일'
        self.ACC_DICT['CalendarOneYear']['group'] = 99

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
            src = self.graph_drawer.convert_to_string(src)
            link = self.graph_drawer.convert_to_string(link)
            trg = self.graph_drawer.convert_to_string(trg)
            self.ACC_DICT[src][link] = trg

    def get_graph(self, sparql_results, show=True):
        net = self.graph_drawer.get_graph(sparql_results, acc_dict=self.ACC_DICT, save=show)
        return net

    def get_nx_graph(self, sparql_results):
        return self.graph_drawer.get_nx_graph(sparql_results, acc_dict=self.ACC_DICT)

    def get_sub_tree_graph(self, sub_tree):
        sparql_results = self.sparql.get_sub_tree_relations(sub_tree)
        self.get_graph(sparql_results, show=True)