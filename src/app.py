from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from rdflib import Graph
from streamlit_agraph import TripleStore, agraph, Config, Set, Node, Triple, Edge
from ontology import OntologySystem
import yaml

main_path = Path().absolute().parent
data_path = main_path / 'data'
with (main_path / 'src' / 'settings.yml').open('r') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
 
x_question = st.text_input("Insert Question Here: ")

st.sidebar.title('Scenario')
option=st.sidebar.selectbox('Select scenario',('1','2','3'))

query_statement = """
SELECT ?s ?p ?o WHERE { 
    VALUES ?s { acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses acc:PropertyPlantAndEquipment acc:NoncurrentAssets acc:CurrentAssets }
    VALUES ?o { acc:CurrentAssets acc:NoncurrentAssets acc:AssetsAbstract }
    ?s ?p ?o .
}
"""

ontology = OntologySystem(
    acc_name_path=data_path / 'AccountName.csv', 
    rdf_path=data_path / 'AccountRDF.xml',
    kwargs_graph_drawer=settings['ontology']['graph_drawer']
)

results = ontology.sparql.query(query_statement)
ontology.get_graph(results)

htmlfile = open("nx.html", 'r', encoding='utf-8')
source_code = htmlfile.read() 
components.html(source_code, height=900, width=900)


# results = graph.query(query_statement)
# store = TripleStore()
# for subj, pred, obj in results:
#     if len(obj.split('#')) > 1:
#         store.add_triple(subj.split('#')[1], pred.split('#')[1], obj.split('#')[1], "")
#     else:
#         store.add_triple(subj.split('#')[1], pred.split('#')[1], obj, "")
# config = Config(
#     width=1000, 
#     height=500, 
#     directed=True,
#     nodeHighlightBehavior=True, 
#     highlightColor="#F7A7A6", # or "blue"
#     collapsible=True,
#     node={'labelProperty':'label'},
#     link={'labelProperty':'label', 'renderLabel': False}
#     # **kwargs e.g. node_size=1000 or node_color="blue"
# ) 

# return_value = agraph(
#     nodes=list(store.getNodes()), 
#     edges=list(store.getEdges()), 
#     config=config
# )