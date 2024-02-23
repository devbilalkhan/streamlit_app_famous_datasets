
# create a function that plots subplots for univariate analysis using plotly
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
    st.title(f'{selected_icon} Data Visualization - {dataset_name}')
    return dataset_name

def plot_univariate(data, feature, target_column):
    # Define the options
    options = ['Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot', 'Density Plot', 'Bar Plot']
    
    # Use Streamlit's selectbox to let the user select an option
    selected_option = st.selectbox('Select a type of univariate analysis', options)

    # Initialize fig to None
    fig = None

    # Perform the selected analysis
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

    # Display the plot if fig has been set
    if fig:
        st.plotly_chart(fig)



def encode_categorical_columns(data_frame, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        data_frame[col] = label_encoder.fit_transform(data_frame[col])
        label_encoders[col] = label_encoder
    return data_frame, label_encoders

def plot_multivariate(data, features, plot_type, target_column=None):
    print("----", features)
    if plot_type == 'Heatmap':
        # Identify categorical columns
        categorical_columns = data[features].select_dtypes(include=['object', 'category']).columns
        # Encode categorical columns if any
        if not categorical_columns.empty:
            data, _ = encode_categorical_columns(data, categorical_columns)
        # Calculate correlation matrix
        corr = data[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        st.pyplot(plt)
    else:
        # For other plot types, handle categorical columns as usual
        if plot_type == 'Scatter Plot':
            fig = px.scatter_matrix(data, dimensions=features, color=target_column, title='Multivariate Scatter Plot')
            st.plotly_chart(fig)
       
def plot_multivariate(data, features, plot_type, target_column=None):
    if plot_type == 'Scatter Plot':
        fig = px.scatter_matrix(data, dimensions=features, color=target_column, title='Multivariate Scatter Plot')
        st.plotly_chart(fig)
  
    elif plot_type == 'Heatmap':
        # encode data
        data, _ = encode_categorical_columns(data, features)
        corr = data[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        st.pyplot(plt)

def get_multivariate_input(data):
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
  dataset_name = display_dataset()
  data = load_data(dataset_name)
    # Get the list of features from the dataframe
  features = data.columns.tolist()

  st.write(f'## Univariate Analysis')
  # Use Streamlit's selectbox to let the user select a feature
  selected_feature = st.selectbox('Select a feature for univariate analysis', features, key='univariate')
   # Add a selectbox for the target column
  target_column = st.selectbox('Select the target column', features, key='target')
  # Call the function with the selected feature

  plot_univariate(data, selected_feature, target_column)
  get_multivariate_input(data)

if __name__ == "__main__":
    main()