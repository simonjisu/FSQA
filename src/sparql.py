from utils import convert_to_string
from rdflib import Graph
from collections import defaultdict

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
        knowledge_query = self.get_predefined_knowledge(knowledge=knowledge)
        sparql_results = self.query(knowledge_query)
        role_dict = defaultdict(list)
        for s, p, o in sparql_results:
            s, p, o = map(convert_to_string, [s, p, o])
            if s == 'CalendarOneYear' or o == 'CalendarOneYear':
                continue
            if s not in role_dict[o]:
                role_dict[o].append(s)
            
        return role_dict