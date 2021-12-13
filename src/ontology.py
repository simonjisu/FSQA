
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import pandas as pd
from pyvis.network import Network
from collections import defaultdict

class Node(object):
    def __init__(self):
        pass


class SparqlHandler(object):
    def __init__(self, rdf_path):
        self.graph = Graph()
        self.graph.load(rdf_path)

    def query(self, query_statement):
        return self.graph.query(query_statement)

class GraphDrawer(object):
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
    
    def get_graph(self, sparql_results, acc_dict):
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
            src_title = f'Statement: {src_fs}\n Type: {src_type}'
            trg_title = f'Statement: {trg_fs}\n Type: {trg_type}'
            net.add_node(
                src_label, src_label, 
                group=int(src_group), size=self.size_dict[int(src_group)], title=src_title
            )
            net.add_node(
                trg_label, trg_label, 
                group=int(trg_group), size=self.size_dict[int(trg_group)], title=trg_title
            )
            net.add_edge(src_label, trg_label, weight=2, titel=link)
        net.show(self.show_file_name)

class OntologySystem(object):
    def __init__(
        self, 
        acc_name_path,
        rdf_path,
        kwargs_graph_drawer
        ):
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
        self.graph_drawer = GraphDrawer(**kwargs_graph_drawer)
        self.sparql = SparqlHandler(rdf_path)

    def get_graph(self, sparql_results):
        return self.graph_drawer.get_graph(sparql_results, acc_dict=self.ACC_DICT)