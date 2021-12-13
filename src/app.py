from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import spacy_streamlit

from nlu import NLU
from dialog import DialogManager
from ontology import OntologySystem
import yaml

main_path = Path().absolute().parent
data_path = main_path / 'data'
with (main_path / 'src' / 'settings.yml').open('r') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

# Modules
nlu_module = NLU()
dialog_manager = DialogManager()
ontology_module = OntologySystem(
    acc_name_path=data_path / 'AccountName.csv', 
    rdf_path=data_path / 'AccountRDF.xml',
    kwargs_graph_drawer=settings['ontology']['graph_drawer']
)

# Sidebar
st.sidebar.title('Graph option')
graph_option = st.sidebar.selectbox('Show Graph', ('scenario', 'income statement', 'balance sheet'))

# Form
with st.form(key='Form'):
    st.write('Select a scenario')
    scenario_option = st.selectbox('Select scenario', (0, 1, 2, 3))
    scenario_questions = {
        0: '',
        1: 'What is our revenue last year?',
        2: 'What is the cost of revenue ratio in last year?',
        3: 'What happens to operating income when cost of revenue increases by 10% this year?'
    }
    x_question = st.text_input("Insert Question Here(not implemented): ")

    if scenario_option == 0:
        sentence = x_question
    else:
        sentence = scenario_questions.get(scenario_option)

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.write('Scenario Question:', sentence)
    nlu_results = nlu_module(sentence=sentence, scenario=scenario_option)
    key_information = dialog_manager.post_process(nlu_results)

    query_statement = """
    SELECT ?s ?p ?o WHERE { 
        VALUES ?s { acc:CashAndCashEquivalents acc:TradeAndOtherCurrentReceivables acc:PrepaidExpenses acc:PropertyPlantAndEquipment acc:NoncurrentAssets acc:CurrentAssets }
        VALUES ?o { acc:CurrentAssets acc:NoncurrentAssets acc:AssetsAbstract }
        ?s ?p ?o .
    }
    """

    results = ontology_module.sparql.query(query_statement)
    ontology_module.get_graph(results)

    html_paths = {
        'scenario': 'nx.html', 
        'income statement': 'is.html', 
        'balance sheet': 'bs.html'
    }
    htmlfile = open(html_paths[graph_option], 'r', encoding='utf-8')
    source_code = htmlfile.read() 
    components.html(source_code, height=700, width=700)

    with st.expander("Results from each module in json"):
        st.write("**NLU module result: **")
        st.write(nlu_results)
        st.write("**Post-process result: **")
        st.write(key_information)

    spacy_streamlit.visualize_ner(nlu_results['doc'], labels=nlu_module.sp_trf.get_pipe('ner').labels)