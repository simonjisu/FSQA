import networkx as nx
import pandas as pd
import streamlit as st
import spacy_streamlit
import sqlparse
import streamlit.components.v1 as components

def write(data_path, modules):
    nlu_module, dialog_manager, ontology_module, database = modules
    # Form
    with st.form(key='Form'):
        st.write('**Select a scenario**')
        st.write('This is a demonstration only to show how does the framework works, so natural languages understanding part is treated as given.')
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
            3: 'What will be our revenue in the 4th quarter?'
        }
        scenario_option = st.selectbox(label='Select scenario', options=scenario_objects)
        scenario_id = scenario_objects[scenario_option]
        sentence = scenario_questions.get(scenario_id)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write('### Scenario Question')
        st.write(f'**{sentence}**')

        nlu_results = nlu_module(sentence=sentence, scenario=scenario_id)
        key_information = dialog_manager.post_process(nlu_results)
        
        context = key_information['context']
        knowledge, account = key_information['account'].split('.')
        quarter = '4Q' if key_information.get('quarter') is None else key_information.get('quarter')
        year = key_information['year']
        
        if context != 'EMB':
            if key_information.get('subject_account'):
                sub_account = {key_information['subject_account'].split('.')[1]: key_information['subject_apply']}
            else:
                sub_account = None
            
            knowledge_query = ontology_module.sparql.get_predefined_knowledge(knowledge=knowledge+'R')
            results = ontology_module.sparql.query(knowledge_query)
            nx_graph = ontology_module.get_nx_graph(results)
            sub_tree = nx.bfs_successors(nx_graph, source=account)
            sparql_results = ontology_module.sparql.get_sub_tree_relations(dict(sub_tree))
            # Get Query Part
            query_statement = ontology_module.get_SQL(sparql_results, account, quarter, year, sub_account)
            query_result = database.query(query_statement)

            ontology_module.get_graph(sparql_results, show=True)
            st.write('Query Result: ')
            st.write(query_result)

            htmlfile = open('nx.html', 'r', encoding='utf-8')
            source_code = htmlfile.read() 
            components.html(source_code, height=600, width=700)
            
        else:
            model_type = key_information.get('model')
            model = ontology_module.embmodels[model_type]
            # show training result 
            df_rev = pd.read_csv(data_path / 'revenues.csv', encoding='utf-8')\
                .set_index('date').rename(columns={'current': 'actual'})
            st.write('### Revenue Graph: From 2016-2Q to 2021-3Q (billion)')
            st.line_chart(df_rev / 1e9)
            X = df_rev.loc['2021-3Q', ['actual']].values
            result = model.predict(X)
            st.write(f'Prediction given previous quarter revenue, revenue of {quarter} will be **{result[0]:,d}** won.')
            st.write('### Model Summary')
            st.write(model.summary())

        st.write('')
        if context != 'EMB':
            st.image(data_path / 'figs' / 'app_knowledge.png')
        else:
            st.image(data_path / 'figs' / 'app_prediction.png')

        with st.expander("Results from each module in json"):
            st.write("**NLU module result: **")
            st.write(nlu_results)
            st.write("**Post-process result: **")
            st.write(key_information)
            if context != 'EMB':
                st.write("**Query result(After run 'get_SQL' function): **")
                query_str = sqlparse.format(database.get_query_string(query_statement), reindent=True, keyword_case='upper')
                st.code(query_str, language='sql')


        # spacy_streamlit.visualize_ner(nlu_results['doc'], labels=nlu_module.sp_trf.get_pipe('ner').labels)
