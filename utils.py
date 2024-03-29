import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st
import pandas as pd
from db.crud import check_and_create_dataset_name, get_dataset_name
import uuid

@st.cache_data
def load_data(data):
    return data

def select_existing_datasets(collection):
    pass

def display_data_overview(data):
    st.write('## Data')
    st.write(data)
    st.write('## Data Description')
    st.write(data.describe())


def display_dataset(collection):

    #dataset_list = [{'dataset_name': 'Iris'}, {'dataset_name': 'Diamonds'}, {'dataset_name': 'Tips'}, {'dataset_name': 'Titanic'}]
    dataset_list = []
    all_ds_names = check_and_create_dataset_name(dataset_list, collection)

    # Sidebar selection for datasets
    dataset_name = st.sidebar.selectbox('Select Dataset or upload one', all_ds_names, key="datasets_01")
 
    # Display the title with the appropriate icon
    return dataset_name


def get_params_random_forest():
    n_estimators = st.slider('RandomForest: Number of trees', 10, 1000, 100)
    max_depth = st.slider('RandomForest: Max depth', 1, 50, 5)
    random_state = st.slider('RandomForest: Random State', 1, 100, 42)
    min_samples_split = st.slider('RandomForest: Min samples split', 2, 10, 2)
    min_samples_leaf = st.slider('RandomForest: Min samples leaf', 1, 10, 1)
    return {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

def get_params_xgboost():
    n_estimators = st.slider('XGBoost: Number of trees', 10, 1000, 100)
    max_depth = st.slider('XGBoost: Max depth', 1, 50, 5)
    learning_rate = st.number_input('XGBoost: Learning rate', min_value=0.0001, max_value=0.9, value=0.01, step=0.0001, format="%.4f")
    return {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

def get_params_lightgbm():
    n_estimators = st.slider('LightGBM: Number of trees', 10, 1000, 100)
    num_leaves = st.slider('LightGBM: Number of leaves', 10, 1000, 31)
   
    learning_rate = st.number_input('XGBoost: Learning rate', min_value=0.0001, max_value=0.9, value=0.01, step=0.0001, format="%.4f")
    return {'n_estimators': n_estimators, 'num_leaves': num_leaves, 'learning_rate': learning_rate}

def get_params_catboost():
    iterations = st.slider('CatBoost: Number of iterations', 10, 1000, 100)
    depth = st.slider('CatBoost: Depth', 1, 10, 5)
    learning_rate = st.number_input('XGBoost: Learning rate', min_value=0.0001, max_value=0.9, value=0.01, step=0.0001, format="%.4f")
    return {'iterations': iterations, 'depth': depth, 'learning_rate': learning_rate}



def get_model_params(model_name):
    
    if model_name == 'RandomForest':
        return get_params_random_forest()
    elif model_name == 'XGBoost':
        return get_params_xgboost()
    elif model_name == 'LightGBM':
        return get_params_lightgbm()
    elif model_name == 'CatBoost':
        return get_params_catboost()
    else:
        st.error("Unknown model selected")
        return None
    

def clean_dataset_name(dataset_name):
    """
    Cleans the dataset name by replacing spaces and hyphens with underscores.

    Parameters:
    dataset_name (str): The name of the dataset to clean.

    Returns:
    str: The cleaned dataset name.
    """
    dataset_name = dataset_name.replace(" ", "_")
    dataset_name = dataset_name.replace("-", "_")
    return dataset_name

def display_welcome_message():
    """
    Displays a welcome message with instructions for the user.
    """
    st.markdown("""
    # Welcome to the Data Exploration Extravaganza! 🎉

    Hi there, data enthusiast! Before we dive into the thrilling world of data analysis, 
    we need to get our hands on some juicy datasets. Here's how you can embark on this adventure:

    1. **Choose from a list** 📚
    - Take a stroll through repository of datasets in the sidebar.

    2. **Be the Data Maestro** 🎼
    - Have some data of your own? Fantastic! You can conduct your symphony by uploading your 
    dataset directly into our system. Just make sure it's in tune (CSV, Excel, etc.) and not 
    heavier than a tuba (up to 200MB, please).

    Please refer to `How to Use` section in the sidebar for more information. 

    _Select or upload your data using the sidebar on the left and let the magic unfold!_
    """)