import networkx as nx
from pathlib import Path
import pandas as pd
import streamlit as st
import spacy_streamlit
import sqlparse
import streamlit.components.v1 as components
from bertviz import head_view

def write(data_path, modules):
    nlu_module, dialog_manager, ontology_module, database = modules
    st.write('# Knowledge Graph')
    st.write('You can check what kinds of account is available.')
    graph_option = st.selectbox('Show Graph', ('income statement', 'balance sheet'))

    html_paths = {
        'income statement': 'is.html', 
        'balance sheet': 'bs.html'
    }

    htmlfile = Path(data_path / html_paths[graph_option]).open('r', encoding='utf-8')
    source_code = htmlfile.read() 
    components.html(source_code, height=800, width=700)

    # Form
    st.write('# DEMO: Key scenarios in board meetings')
    st.write('Thers are three reference key scenarios by following table, you can copy and phased it or modify a little bit from them')
    st.write("""
    ### What-is?

    In 'What-is' questions, users might want to ask past information based on the fact and knowledge 
    on financial statements. Some ratios which can be calculate from the statements are also important to analyze company's current 
    status.

    - For example, "What was the cost of sales ratio in the last year?" 

    ### What-if:based on fact
    In 'What-if:based on fact' questions, users might want to ask the effect to a certain account 
    by changing another account value. 
    
    - For example, "What happens to the operating income when the cost of sales increases by 10% in the last year?"

    ### What-if:forecasting

    In 'What-if:forecasting' questions, users might want to know future values of a account.

    - For example, "What will be our revenue in 4th quarter?".

    """)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write('## Available Income Statement Accounts')
    #     st.selectbox('', options=dialog_manager.is_account_names)
        
    # with col2:
    #     st.write('## Available Balance Sheet Accounts')
    #     st.selectbox('', options=dialog_manager.bs_account_names)
    st.write('## Try your self')
    with st.form(key='Form'): 
        sentence = st.text_input('Insert a questions')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write('## Scenario Question')
        st.write(f'**{sentence}**')
        nlu_results = nlu_module(text=sentence)
        error, key_information = dialog_manager.post_process(nlu_results)
        # st.write(nlu_results)
        # st.write(key_information)
        if error:
            st.error(key_information)

        spacy_streamlit.visualize_ner(nlu_results['doc'], labels=['APPLY', 'IS', 'BS', 'PERCENT', 'TIME'])

        if key_information['intent'] is not None:
            intent = key_information['intent']
            knowledge, account = key_information['target_account'].split('.')
            year = key_information['year']
            quarter = key_information['quarter']

            if intent != 'IF.forecast':
                if key_information.get('subject_account'):
                    sub_account = {key_information['subject_account'].split('.')[1]: key_information['subject_apply']}
                else:
                    sub_account = None
                
                # Get drawings
                knowledge_query = ontology_module.sparql.get_predefined_knowledge(knowledge=knowledge+'R')
                results = ontology_module.sparql.query(knowledge_query)
                nx_graph = ontology_module.get_nx_graph(results)
                sub_tree = nx.bfs_successors(nx_graph, source=account)
                sparql_results = list(ontology_module.sparql.get_sub_tree_relations(dict(sub_tree)))
                # Get Query Part
                query_statement = ontology_module.get_SQL(sparql_results, account, quarter, year, sub_account)
                query_result = database.query(query_statement)

                ontology_module.get_graph(sparql_results, show=True)
                st.write('Query Result: ')
                st.write(query_result)
                if sparql_results:
                    htmlfile = open(ontology_module.graph_drawer.show_file_name, 'r', encoding='utf-8')
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

        with st.expander("Results from each module in json"):
            st.write("**NLU module result: **")
            st.write(nlu_results)
            st.write("**Post-process result: **")
            st.write(key_information)
            if intent != 'IF.forecast':
                st.write("**Query result(After run 'get_SQL' function): **")
                query_str = sqlparse.format(database.get_query_string(query_statement), reindent=True, keyword_case='upper')
                st.code(query_str, language='sql')

        
