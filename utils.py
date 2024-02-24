import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st

def set_color_map(color_list):
    cmap_custom = ListedColormap(color_list)
    print("Notebook Color Schema:")
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