import streamlit as st
import streamlit.components.v1 as components

def write(data_path, modules):
    # Sidebar
    st.title('Predefined Knowledge Graph')
    graph_option = st.selectbox('Show Graph', ('income statement', 'balance sheet'))

    html_paths = {
        'income statement': 'is.html', 
        'balance sheet': 'bs.html'
    }

    htmlfile = open(html_paths[graph_option], 'r', encoding='utf-8')
    source_code = htmlfile.read() 
    components.html(source_code, height=700, width=700)

