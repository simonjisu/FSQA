from pathlib import Path
import streamlit as st
from rdflib import Graph
from streamlit_agraph import TripleStore, agraph, Config, Set, Node, Triple, Edge

main_path = Path().absolute().parent
data_path = main_path / 'data'

x_question = st.text_input("Insert Question Here: ")

class TripleStore:
    def __init__(self) ->None:
        self.nodes_set: Set[Node] = set()
        self.edges_set: Set[Edge] = set()
        self.triples_set: Set[Triple] = set()

    def add_triple(self, node1, link, node2, picture=""):
        nodeA = Node(node1, svg=picture)
        nodeB = Node(node2)
        edge = Edge(source=nodeA.id, target=nodeB.id, label=link, renderLabel=True)  # linkValue=link
        triple = Triple(nodeA, edge, nodeB)
        self.nodes_set.update([nodeA, nodeB])
        self.edges_set.add(edge)
        self.triples_set.add(triple)

    def getTriples(self)->Set[Triple]:
        return self.triples_set

    def getNodes(self)->Set[Node]:
        return self.nodes_set

    def getEdges(self)->Set[Edge]:
        return self.edges_set

# filter
def get_subgraph(graph, query_statement):
    results = graph.query(query_statement)
    store = TripleStore()

    for subj, pred, obj in results:
        store.add_triple(subj.split('#')[1], pred.split('#')[1], obj.split('#')[1], "")
    return store

query_statement = """
CONSTRUCT {
    ?s acc:partOf ?o ;
}
WHERE {
    ?s acc:Account_Belonging acc:BalanceSheet .
    ?o acc:Account_Belonging acc:BalanceSheet .
    FILTER (?s != ?o)
}
"""
query_statement = """
CONSTRUCT {
    ?s acc:partOf acc:CurrentAsset ;
}
WHERE {
    ?s acc:Account_Belonging acc:BalanceSheet .
    ?o acc:Account_Belonging acc:BalanceSheet .
    FILTER NOT EXISTS {
        ?s acc:partOf ?o
    }
}
"""
graph = Graph()
graph.load(data_path / 'AccountRDF.xml')
results = graph.query(query_statement)
store = TripleStore()
for subj, pred, obj in results:
    if len(obj.split('#')) > 1:
        store.add_triple(subj.split('#')[1], pred.split('#')[1], obj.split('#')[1], "")
    else:
        store.add_triple(subj.split('#')[1], pred.split('#')[1], obj, "")
config = Config(
    width=1000, 
    height=500, 
    directed=True,
    nodeHighlightBehavior=True, 
    highlightColor="#F7A7A6", # or "blue"
    collapsible=True,
    node={'labelProperty':'label'},
    link={'labelProperty':'label', 'renderLabel': False}
    # **kwargs e.g. node_size=1000 or node_color="blue"
) 

return_value = agraph(
    nodes=list(store.getNodes()), 
    edges=list(store.getEdges()), 
    config=config
)