from pathlib import Path
import pygraphviz as pgv
import streamlit as st
from rdflib import Graph
from streamlit_agraph import TripleStore, agraph, Config

main_path = Path().absolute().parent
data_path = main_path / 'data'

x_question = st.text_input("Insert Question Here: ")

# filter
def get_subgraph(graph, query_statement):
    results = graph.query(query_statement)
    store = TripleStore()

    for subj, pred, obj in results.graph:
        store.add_triple(subj, pred, obj, "")
    return store

query_string = """
CONSTRUCT {
    ?acc acc:partOf ?acc .         
}
WHERE {
    ?acc acc:Account_Belonging acc:BalanceSheet .
    ?acc rdfs:label ?any .
}
"""
graph = Graph()
graph.load(data_path / 'AccountRDF.xml')
store = get_subgraph(graph, query_string)

config = Config(
    width=1000, 
    height=1000, 
    directed=True,
    nodeHighlightBehavior=True, 
    highlightColor="#F7A7A6", # or "blue"
    collapsible=True,
    node={'labelProperty':'label'},
    link={'labelProperty':'label', 'renderLabel': True}
    # **kwargs e.g. node_size=1000 or node_color="blue"
) 

return_value = agraph(
    nodes=list(store.getNodes()), 
    edges=list(store.getEdges()), 
    config=config
)