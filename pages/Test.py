# Data_Cleaning.py
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from utils import load_data, display_data_overview,\
         clean_dataset_name, display_welcome_message,\
         select_existing_datasets
from sklearn.preprocessing import StandardScaler
from db.crud import delete_dataset_records, check_and_create_single_ds_name,\
                     get_dataset_name
from config import DATASET_COLLECTION_NAME, ROWS





def preprocess_dataset(data):

    return data



def preprocess_data(dataset_option):
    """
    Preprocesses the data by handling missing values and scaling the data.
    """
    data = None
    placeholder_success = st.empty()
    #delete_dataset_records('cvd', DATASET_COLLECTION_NAME)
    st.write(get_dataset_name(DATASET_COLLECTION_NAME))
    
    if dataset_option == 'Select Existing Dataset':
        dataset_name = select_existing_datasets('dataset_names')
        data = load_data(dataset_name)     
       
        st.write(f"Loaded dataset: {dataset_name}")
        st.write(data.head())
        # Process the data from the file path or database reference here

    elif dataset_option == 'Upload New Dataset':
        st.sidebar.write('Upload is temporary unavailable. Please select an existing dataset.')
        # result = handle_file_upload()      
        # if result is not None:
        #     data, dataset_name = result         
        #     st.session_state["dataset"] = data
        #     placeholder_success.success(f"Dataset {dataset_name} saved successfully!") 

    if data is not None:
        # Get a list of all columns
        columns = [column for column in data.columns]
        
        # Select the target variable
        st.write('### Select the Target Variable')
        target_column = st.selectbox('Select the target column', data.columns)

        if target_column is not None:
        #st.write(f'#### Target column selected: {target_column}')
            st.markdown(f'##### Target column selected:  <span style="background-color: rgba(128, 0, 0, 0.5); padding: 5px; border-radius: 5px;"> {target_column}</span>', unsafe_allow_html=True)
            target_data = data[target_column]
            st.session_state['target'] = target_column

        # Handle the target variable and recreate the DataFrame
        data = handle_target_variable(data, target_column)
        data = pd.DataFrame(data, columns=columns)

        # Separate the target column data and drop it from the main data
        target_column_data = data[target_column]
        data.drop(target_column, axis=1, inplace=True)
        cols_without_target = [col for col in data.columns]
       
        read_only_data = pd.concat([data, target_data], axis=1)
        display_data_overview(read_only_data)
        display_categorical_columns(read_only_data)
        display_most_frequent_values(read_only_data)
        display_unique_values(read_only_data)
        display_missing_values(read_only_data, dataset_name)

        st.markdown('---')
        if st.session_state['temp'] is not None:
            data = st.session_state['temp']

        data = drop_columns(data)
        cols_without_target = data.columns

        # If there are missing values, handle them
        if data.isnull().sum().any() > 0:
            st.markdown('---')
            #handle_missing_values(data, dataset_name)
            st.write('## Missing Values')
            plot_missing_values(data)
            
            st.session_state['temp'] = select_imputation_strategy(data)
            data = st.session_state['temp']          
            st.markdown('---')
            data = pd.DataFrame(data, columns=cols_without_target)
        
        # If there are categorical columns, convert them to numerical
        if pd.DataFrame(data).select_dtypes(include=['object', 'category']).shape[1] > 0:        
            data = convert_to_numerical(data)

        st.write('## Outlier Removal')
        if st.selectbox('Remove Outliers', ['No', 'Yes'], key="scale_data") == 'Yes': 
            data = remove_outliers(data)      

        # Scale the data if 
        st.write('## Scale Data')
        scale_option = st.selectbox('Do you want to scale the data?', ('No', 'Yes'), key='sc_01')
        features_data_scaled = None
        is_scaled = False
        if scale_option == 'Yes':
            features_data_scaled, is_scaled = scale_data(data, scale_option) 
            data = pd.DataFrame(features_data_scaled, columns=cols_without_target)
         
       
        
        #Add the target column back to the data
            data[target_column] = target_column_data
            store_dataset(data)
            
    else:
        display_welcome_message()           
    placeholder_success.empty()
    return data

# session state for presisting datasets
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None
if 'temp' not in st.session_state:
    st.session_state['temp'] = None

def store_dataset(data):
    st.session_state['dataset'] = data
    return data

# Main


if __name__ == "__main__":
  main()