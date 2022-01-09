from pathlib import Path
import streamlit as st

from nlu import NLUModule
from dialog import DialogManager
from ontology import OntologySystem
from db import DBHandler
import yaml
import pages.demo
import pages.graph

PAGES = {
    'Demo': pages.demo,
    # 'Predefined Graph': pages.graph,
}

# Modules
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def create_modules(data_path, settings):

    nlu_settings = settings['nlu']
    nlu_module = NLUModule(
        checkpoint_path=data_path / nlu_settings['checkpoint_file'],
        labels_path=data_path / nlu_settings['labels_file']
    )
    dialog_settings = settings['dialog']
    dialog_manager = DialogManager(
        words_path=data_path / dialog_settings['words_file'],
        acc_name_path=data_path / dialog_settings['acc_name_file'],
        rdf_path=data_path / dialog_settings['rdf_file']
    )

    ontology_settings = settings['ontology']
    ontology_module = OntologySystem(
        acc_name_path=data_path / ontology_settings['acc_name_file'], 
        rdf_path=data_path / ontology_settings['rdf_file'],
        model_path=data_path / ontology_settings['model']['model_name'],
        kwargs_graph_drawer=ontology_settings['graph_drawer']
    )
    database = DBHandler(settings['db'])
    return nlu_module, dialog_manager, ontology_module, database
    
def main():
    st.set_page_config(
        page_title="Demo for FSQA", 
        page_icon=":bulb:",
        layout= 'wide', # centered / "wide",
        initial_sidebar_state="collapsed"
    )
    # st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    main_path = Path().absolute().parent
    data_path = main_path / 'src' / 'data'
    setting_path = main_path / 'setting_files'
    with (setting_path / 'app_settings.yml').open('r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    if selection == 'Demo':
        modules = create_modules(data_path, settings)
    else:
        modules = None
    with st.spinner(f"Loading {selection} ..."):
        page.write(data_path, modules)

if __name__ == '__main__':
    main()
