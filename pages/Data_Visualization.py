# Data_Visualization.py
# create a function that plots subplots for univariate analysis using plotly
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import load_data, clean_dataset_name, select_existing_datasets

def load_data(dataset_name):
    """
    Loads the specified dataset.

    Parameters:
    dataset_name (str): The name of the dataset to load.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """
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

def display_dataset():
    """
    Displays the dataset selection in the sidebar and sets the target variable based on the selected dataset.

    Parameters:
    None

    Returns:
    str: The name of the selected dataset.
    """
    dataset_icons = {
    'Iris': '🌸',
    'Diamonds': '💎',
    'Tips': '💲',
    'Titanic': '🚢'
    }

    dataset_name = st.sidebar.selectbox('Select Dataset', list(dataset_icons.keys()))
    if dataset_name == 'Iris':
        st.session_state['target_variable'] = 'species'
    elif dataset_name == 'Diamonds':
        st.session_state['target_variable'] = 'price'
    elif dataset_name == 'Tips':
        st.session_state['target_variable'] = 'tip'
    elif dataset_name == 'Titanic':
        st.session_state['target_variable'] = 'survived'
    selected_icon = dataset_icons.get(dataset_name, '')

    st.title(f'{selected_icon} Dataset')
    return dataset_name


def plot_univariate(data, feature, target_column):
    """
    Plots univariate analysis based on the selected option.

    Parameters:
    data (pandas.DataFrame): The DataFrame to plot.
    feature (str): The feature to plot.
    target_column (str): The target column.

    Returns:
    None
    """
    options = ['Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot', 'Density Plot', 'Bar Plot']
    selected_option = st.selectbox('Select a type of univariate analysis', options)

    fig = None

    if selected_option == 'Scatter Plot':
        fig = px.scatter(data, x=feature, y=target_column, title=f'Scatter Plot of {feature} vs {target_column}')
    elif selected_option == 'Histogram':
        fig = px.histogram(data, x=feature, title=f'Univariate Analysis of {feature}')
    elif selected_option == 'Box Plot':
        fig = px.box(data, y=feature, title=f'Box Plot of {feature}')
    elif selected_option == 'Violin Plot':
        fig = px.violin(data, y=feature, title=f'Violin Plot of {feature}')
    elif selected_option == 'Density Plot':
        fig = px.density_contour(data, x=feature, y=target_column, title=f'Density Plot of {feature} vs {target_column}')
    elif selected_option == 'Bar Plot':
        fig = px.bar(data, x=feature, y=target_column, title=f'Bar Plot of {feature} vs {target_column}')

    if fig:
        st.plotly_chart(fig)


def encode_categorical_columns(data_frame, categorical_columns):
    """
    Encodes categorical columns in the DataFrame using LabelEncoder.

    Parameters:
    data_frame (pandas.DataFrame): The DataFrame to encode.
    categorical_columns (list): The list of categorical columns to encode.

    Returns:
    pandas.DataFrame: The encoded DataFrame.
    dict: The dictionary of LabelEncoders used for encoding.
    """
    label_encoders = {}
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        data_frame[col] = label_encoder.fit_transform(data_frame[col])
        label_encoders[col] = label_encoder
    return data_frame, label_encoders


def plot_multivariate(data, features, plot_type, target_column=None):
    """
    Plots multivariate analysis based on the selected option.

    Parameters:
    data (pandas.DataFrame): The DataFrame to plot.
    features (list): The list of features to plot.
    plot_type (str): The type of plot to create.
    target_column (str, optional): The target column. Defaults to None.

    Returns:
    None
    """
    if plot_type == 'Heatmap':
        categorical_columns = data[features].select_dtypes(include=['object', 'category']).columns
        if not categorical_columns.empty:
            data, _ = encode_categorical_columns(data, categorical_columns)
        corr = data[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        st.pyplot(plt)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter_matrix(data, dimensions=features, color=target_column, title='Multivariate Scatter Plot')
        st.plotly_chart(fig)

def plot_multivariate(data, features, plot_type, target_column=None):
    """
    Plots multivariate analysis based on the selected option.

    Parameters:
    data (pandas.DataFrame): The DataFrame to plot.
    features (list): The list of features to plot.
    plot_type (str): The type of plot to create.
    target_column (str, optional): The target column. Defaults to None.

    Returns:
    None
    """
    if plot_type == 'Scatter Plot':
        fig = px.scatter_matrix(data, dimensions=features, color=target_column, title='Multivariate Scatter Plot')
        st.plotly_chart(fig)
    elif plot_type == 'Heatmap':
        data, _ = encode_categorical_columns(data, features)
        corr = data[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        st.pyplot(plt)

def get_multivariate_input(data):
    """
    Gets the input for multivariate analysis from the user.

    Parameters:
    data (pandas.DataFrame): The DataFrame to get input for.

    Returns:
    None
    """
    st.subheader('Multivariate Analysis')
    features = data.columns.tolist()
    selected_features = st.multiselect('Select features for multivariate analysis', features, default=features[:3])
    plot_type_options = ['Scatter Plot',  'Heatmap']
    plot_type = st.selectbox('Select plot type for multivariate analysis', plot_type_options, key='multivariate')
    target_column = None
    if plot_type in ['Scatter Plot', 'Pair Plot']:
        target_column = st.selectbox('Select the target column', features)
    if len(selected_features) >= 2:
        plot_multivariate(data, selected_features, plot_type, target_column)
    else:
        st.warning("Please select at least two features for multivariate analysis.")


def main():
    # load datasets 
    # Set the title of the page
    st.title(' Data Visualization 📊')
    # Display a message
    st.write('This is the Model Training page. Feed me data!!')   
    dataset_name = select_existing_datasets('dataset_names')
    dataset_name = clean_dataset_name(dataset_name)
    data = load_data(dataset_name)
    
    # if data is none then load the data from data folder csv file
    if data is None:
        # strip the datanames from any whitespace and hyphens and replave with underscores       
        dataset_name = dataset_name.lower()
        data = pd.read_csv(f'data/{dataset_name}.csv')
    # Get the list of features from the dataframe
    features = data.columns.tolist()

    st.write(f'## Univariate Analysis')
    # Use Streamlit's selectbox to let the user select a feature
    selected_feature = st.selectbox('Select a feature for univariate analysis', features, key='univariate')
    # Add a selectbox for the target column
    target_column = st.selectbox('Select the target column', features, key='target')  

    # Plot analysis
    plot_univariate(data, selected_feature, target_column)
    get_multivariate_input(data)

if __name__ == "__main__":
    main()