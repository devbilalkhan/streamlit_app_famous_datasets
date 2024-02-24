import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st

def set_color_map(color_list):
    cmap_custom = ListedColormap(color_list)
    
    sns.palplot(sns.color_palette(color_list))
    plt.show()
    return cmap_custom
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
cmap_custom = set_color_map(color_list)

#Load the datasets
def load_data(dataset_name):
    if dataset_name == 'Iris':
        data = sns.load_dataset('iris')
    elif dataset_name == 'Diamonds':
        data = sns.load_dataset('diamonds')
    elif dataset_name == 'Titanic':
        data = sns.load_dataset('titanic')  
    elif dataset_name == 'Tips':
        data = sns.load_dataset('tips')
    else:
        data = None
    return data

def display_data_overview(data):
    st.write('## Data')
    st.write(data)
    st.write('## Data Description')
    st.write(data.describe())

def display_dataset():

    dataset_icons = {
    'Iris': '🌸',
    'Diamonds': '💎',
    'Tips': '💲',
    'Titanic': '🚢'
    }

    # Sidebar selection for datasets
    dataset_name = st.sidebar.selectbox('Select Dataset', list(dataset_icons.keys()))
    if dataset_name == 'Iris':
        st.session_state['target_variable'] = 'species'
    elif dataset_name == 'Diamonds':
        st.session_state['target_variable'] = 'price'
    elif dataset_name == 'Tips':
        st.session_state['target_variable'] = 'tip'
    elif dataset_name == 'Titanic':
        st.session_state['target_variable'] = 'survived'
    # Get the corresponding icon for the selected dataset
    selected_icon = dataset_icons.get(dataset_name, '')

    # Display the title with the appropriate icon
    st.title(f'{selected_icon} Data Cleaning - {dataset_name}')
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