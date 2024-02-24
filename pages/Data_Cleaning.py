# Data_Cleaning.py
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from utils import set_color_map
from sklearn.model_selection import train_test_split
from utils import load_data, display_data_overview, display_dataset
from sklearn.preprocessing import StandardScaler

def display_categorical_columns(data):
    """
    Displays the categorical columns in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the categorical columns from.
    """
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    st.write('## Categorical Columns')
    cat_columns_df = pd.DataFrame(cat_columns, columns=['Categorical Features'])
    st.write(cat_columns_df)

def display_most_frequent_values(data):
    """
    Displays the most frequent values in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the most frequent values from.
    """
    st.write('## Most Frequent Values')
    most_frequent_values(data)

def display_unique_values(data):
    """
    Displays the unique values in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the unique values from.
    """
    st.write('## Unique Values')
    unique_values(data)

def display_missing_values(data, name):
    """
    Displays the missing values in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the missing values from.
    name (str): The name of the DataFrame.
    """
    st.write('## Missing Values')
    total = data.isnull().sum()
    if total.sum() == 0:
        st.write('No missing values found.')
    else:
        percent = (data.isnull().sum()/data.isnull().count()*100)
        tt = pd.concat([total, round(percent,2)], axis=1, keys=['Total', 'Percent'])
        types = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            types.append(dtype)
        tt['Types'] = types
        tt['Percent'] = tt['Percent'].astype(str) + '%'
        tt = tt[tt['Percent'] != '0.0%']
        st.write(np.transpose(tt))
        st.write('Total missing values: ', total.sum())

def most_frequent_values(data):
    """
    Displays the most frequent values in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the most frequent values from.
    """
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    st.write(np.transpose(tt))


def unique_values(data):
    """
    Displays the unique values in the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display the unique values from.
    """
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    st.write(np.transpose(tt))

def convert_to_numerical(data, target_column=None):
    """
    Converts categorical and object columns to numerical.

    Parameters:
    data (pandas.DataFrame): The DataFrame to convert.
    target_column (str, optional): The target column. Defaults to None.

    Returns:
    pandas.DataFrame: The converted DataFrame.
    """
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    cat_columns = list(cat_columns)
    columns = data.columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), cat_columns),
        ],
        remainder='passthrough'
    )
    data_features = column_transformer.fit_transform(data)
    data_transformed = pd.DataFrame(data_features, columns=columns)
    return data_transformed

def impute_missing_values(data, strategy='mean'):
    """
    Imputes missing values in the data using a specified strategy.

    Parameters:
    data (pandas.DataFrame): The DataFrame to impute missing values in.
    strategy (str, optional): The imputation strategy. Defaults to 'mean'.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    data = convert_to_numerical(data)
    imputer = SimpleImputer(strategy=strategy)
    data = imputer.fit_transform(data)
    return data

def impute_missing_values_knn(data, n_neighbors=5):
    """
    Imputes missing values in the data using K-Nearest Neighbors.

    Parameters:
    data (pandas.DataFrame): The DataFrame to impute missing values in.
    n_neighbors (int, optional): The number of neighbors to use for KNN imputation. Defaults to 5.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    data = convert_to_numerical(data)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data = imputer.fit_transform(data)
    return data

def remove_columns(data, columns):
    """
    Removes specified columns from the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to remove columns from.
    columns (list): The list of columns to remove.

    Returns:
    pandas.DataFrame: The DataFrame with the specified columns removed.
    """
    data = data.drop(columns=columns)
    return data

def handle_missing_values(data, dataset_name):
    """
    Handles missing values in the data based on the dataset name.

    Parameters:
    data (pandas.DataFrame): The DataFrame to handle missing values in.
    dataset_name (str): The name of the dataset.
    """
    if dataset_name == 'Titanic':
        st.write('## Handle Missing Values')

def drop_columns_with_missing_values(data):
    """
    Drops columns with missing values from the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to drop columns from.

    Returns:
    pandas.DataFrame: The DataFrame with columns with missing values dropped.
    """
    st.write('### Drop Columns with Missing Values')
    if st.checkbox('Select to drop columns'):
        columns_to_drop = st.multiselect('Select columns to drop', data.columns[data.isnull().any()])
        if columns_to_drop:
            data = remove_columns(data, columns_to_drop)
            st.write('Columns dropped:', columns_to_drop)
    return data 

def select_imputation_strategy(data):
    """
    Allows the user to select an imputation strategy for missing values.

    Parameters:
    data (pandas.DataFrame): The DataFrame to impute missing values in.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    st.write('### Impute Missing Values')
    impute_options = st.selectbox('Choose an imputation method', 
                                    ['None', 
                                    'Impute with mean (for numerical columns)', 
                                    'Impute with median (for numerical columns)',
                                    'Impute with mode (for categorical columns)',
                                    'Impute with KNN'])

    if 'Impute with' in impute_options:
        data = apply_imputation(data, impute_options)
    return data 

def display_data_after_missing_values(data):
    """
    Displays the data after handling missing values.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display.
    """
    st.write('### Data after Handling Missing Values')
    st.write(pd.DataFrame(data).head())

def apply_imputation(data, impute_options):
    """
    Applies the selected imputation strategy to the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to impute missing values in.
    impute_options (str): The selected imputation strategy.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    if 'mean' in impute_options:
        strategy = 'mean'
    elif 'median' in impute_options:
        strategy = 'median'
    elif 'mode' in impute_options:
        strategy = 'most_frequent'
    
    if 'KNN' in impute_options:
        n_neighbors = st.number_input('Number of neighbors for KNN', min_value=1, value=5)
        data = impute_missing_values_knn(data, n_neighbors=n_neighbors)
    else:
        data = impute_missing_values(data, strategy=strategy)   
  
    return data

def scale_data(data, key):
    """
    Scales the data if the user selects to do so.

    Parameters:
    data (pandas.DataFrame): The DataFrame to scale.
    key (str): The key for the Streamlit widget.

    Returns:
    pandas.DataFrame, bool: The potentially scaled DataFrame and a flag indicating if scaling was applied.
    """
    st.write('## Data Scaling')
    is_scaled = False
    scale_option = st.selectbox('Do you want to scale the data?', ('No', 'Yes'), key=key)

    if scale_option == 'Yes':     
        st.write('## Data after Scaling')
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        st.write(pd.DataFrame(data).head())
        is_scaled = True
    else:
        st.write('## Data without Scaling')
        st.write(pd.DataFrame(data).head())
        is_scaled = False

    return data, is_scaled

def initialize_session_state():
    """
    Initializes the session state.
    """
    if 'target_variable' not in st.session_state:
        st.session_state['target_variable'] = None

def encode_target(data, target_column):
    """
    Encodes the target column if it is categorical.

    Parameters:
    data (pandas.DataFrame): The DataFrame to encode the target column in.
    target_column (str): The target column to encode.

    Returns:
    pandas.DataFrame, LabelEncoder: The DataFrame with the encoded target column and the LabelEncoder used.
    """
    if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
    return data, label_encoder

def inverse_encode(data, target_column):
    """
    Inversely encodes the target column if it was previously encoded.

    Parameters:
    data (pandas.DataFrame): The DataFrame to inversely encode the target column in.
    target_column (str): The target column to inversely encode.

    Returns:
    str: The target column.
    """
    if st.session_state['is_encoded']:
        data[target_column] = st.session_state['label_encoder'].inverse_transform(data[target_column])
    return target_column

def handle_target_variable(data, target_column):
    """
    Handles the target variable by encoding it if it is categorical.

    Parameters:
    data (pandas.DataFrame): The DataFrame to handle the target variable in.
    target_column (str): The target variable to handle.

    Returns:
    pandas.DataFrame: The DataFrame with the handled target variable.
    """
    if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
        st.write('### Target Variable Value Counts')
        st.write(data[target_column].value_counts())              
          
        data, le = encode_target(data, target_column)
        st.session_state['is_encoded'] = True
        st.session_state['label_encoder'] = le
          
    return data

# Main
def main():
        # Display the dataset and load the data
    dataset_name = display_dataset()
    data = load_data(dataset_name)

    # Get a list of all columns
    columns = [column for column in data.columns]

    # Select the target variable
    st.write('### Select the Target Variable')
    target_column = st.selectbox('Select the target column', data.columns)
    target_data = data[target_column]

    # Handle the target variable and recreate the DataFrame
    data = handle_target_variable(data, target_column)    
    data = pd.DataFrame(data, columns=columns)

    # Separate the target column data and drop it from the main data
    target_column_data = data[target_column]
    data.drop(target_column, axis=1, inplace=True)
    cols_without_target = [col for col in data.columns]

    # If data is not None, display various data overviews
    if data is not None:
        read_only_data = pd.concat([data, target_data], axis=1)
        display_data_overview(read_only_data)
        display_categorical_columns(read_only_data)
        display_most_frequent_values(read_only_data)
        display_unique_values(read_only_data)
        display_missing_values(read_only_data, dataset_name)

        # If there are missing values, handle them
        if data.isnull().sum().any() > 0:
            st.markdown('---')
            handle_missing_values(data, dataset_name)
            data = drop_columns_with_missing_values(data)
            cols_without_target = data.columns
            data = select_imputation_strategy(data)
            display_data_after_missing_values(data)

        # If there are categorical columns, convert them to numerical
        if pd.DataFrame(data).select_dtypes(include=['object', 'category']).shape[1] > 0:        
            data = convert_to_numerical(data)
        st.markdown('---')

        # Scale the data if selected
        features_data_scaled, is_scaled = scale_data(data, key='sc_01') 
        data = pd.DataFrame(features_data_scaled, columns=cols_without_target)

        # Add the target column back to the data
        data[target_column] = target_column_data

        # If the data was scaled, save it to a CSV file
        if is_scaled:   
            data.to_csv(f'data/data_{dataset_name}.csv', index=False)
if __name__ == "__main__":
    main()
