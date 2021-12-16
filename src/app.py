from pathlib import Path
import streamlit as st

from nlu import NLU
from dialog import DialogManager
from ontology import OntologySystem
from db import DBHandler
import yaml
import pages.demo
import pages.graph

PAGES = {
    'Predefined Graph': pages.graph,
    'Demo': pages.demo
}

# Modules
def create_modules(data_path, settings):
    nlu_module = NLU()
    dialog_manager = DialogManager()
    ontology_module = OntologySystem(
        acc_name_path=data_path / 'AccountName.csv', 
        rdf_path=data_path / 'AccountRDF.xml',
        model_path=data_path / settings['ontology']['model']['model_name'],
        kwargs_graph_drawer=settings['ontology']['graph_drawer']
    )
    database = DBHandler(settings['db'])
    return nlu_module, dialog_manager, ontology_module, database
    
def main():
    st.set_page_config(
        page_title="Demo for FSQA", 
        page_icon=":bulb:",
        layout="centered", # "wide",
        # initial_sidebar_state="collapsed"
    )
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    main_path = Path().absolute().parent
    data_path = main_path / 'data'
    with (main_path / 'src' / 'settings.yml').open('r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    if selection == 'Demo':
        modules = create_modules(data_path, settings)
    else:
        modules = None
    with st.spinner(f"Loading {selection} ..."):
        page.write(data_path, modules)

if __name__ == '__main__':
    main()
