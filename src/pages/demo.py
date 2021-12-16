import networkx as nx
import pandas as pd
import streamlit as st
import spacy_streamlit
import streamlit.components.v1 as components

def write(data_path, modules):
    nlu_module, dialog_manager, ontology_module = modules
    # Form
    with st.form(key='Form'):
        st.write('Select a scenario')
        scenario_objects = {
            'none': 0, 
            'Asking information based on fact and knowledge': 1, 
            'What if: Analysis based on fact': 2,
            'What if: Forecasting with embedded ML': 3
        }
        scenario_questions = {
            0: '',
            1: 'What is the Cost of Sales Ratio in last year?',
            2: 'What happens to the Operating Income when the Cost of Sales increases by 10% this year?',
            3: 'What will be our revenue in 4th quarter?'
        }
        scenario_option = st.selectbox(label='Select scenario', options=scenario_objects)
        scenario_id = scenario_objects[scenario_option]
        sentence = scenario_questions.get(scenario_id)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write('Scenario Question:', sentence)
        nlu_results = nlu_module(sentence=sentence, scenario=scenario_id)
        key_information = dialog_manager.post_process(nlu_results)
        
        knowledge, account = key_information['account'].split('.')
        knowledge_query = ontology_module.sparql.get_predefined_knowledge(knowledge=knowledge+'R')
        results = ontology_module.sparql.query(knowledge_query)
        nx_graph = ontology_module.get_nx_graph(results)
        sub_tree = nx.bfs_successors(nx_graph, source=account)

        # TODO: Get Query Part
        

        ontology_module.get_sub_tree_graph(sub_tree)

        if key_information['context'] != 'EMB':
            htmlfile = open('nx.html', 'r', encoding='utf-8')
            source_code = htmlfile.read() 
            components.html(source_code, height=700, width=700)
        else:
            df_rev = pd.read_csv(data_path / 'train_data.csv', encoding='utf-8').rename(
                columns={'value': 'Revenue'})
            df_rev['Date'] = df_rev['bsns_year'].astype(str) + '-' + df_rev['quarter']
            
            st.write('### Revenue Graph: From 2016-1Q to 2021-3Q (billion)')
            st.line_chart(df_rev.loc[:, ['Date', 'Revenue']].set_index('Date') / 1e9)

            # TODO: add machine learning prediction


        with st.expander("Results from each module in json"):
            st.write("**NLU module result: **")
            st.write(nlu_results)
            st.write("**Post-process result: **")
            st.write(key_information)

        spacy_streamlit.visualize_ner(nlu_results['doc'], labels=nlu_module.sp_trf.get_pipe('ner').labels)
