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
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    st.write('## Categorical Columns')
    cat_columns_df = pd.DataFrame(cat_columns, columns=['Categorical Features'])
    st.write(cat_columns_df)

def display_most_frequent_values(data):
    st.write('## Most Frequent Values')
    most_frequent_values(data)

def display_unique_values(data):
    st.write('## Unique Values')
    unique_values(data)

def display_missing_values(data, name):
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

        # Add percent symbol to 'Percent' column
        tt['Percent'] = tt['Percent'].astype(str) + '%'

        # Filter rows where 'Percent' is greater than zero
        tt = tt[tt['Percent'] != '0.0%']

        st.write(np.transpose(tt))
        st.write('Total missing values: ', total.sum())

def most_frequent_values(data):
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
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    st.write(np.transpose(tt))


# create function to impute missing values but first have function that converts categorical and object to numerical
def convert_to_numerical(data, target_column=None):
    # Convert categorical and object columns to numerical
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    cat_columns = list(cat_columns)


    columns = data.columns

    # Create a ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), cat_columns),
        ],
        remainder='passthrough'
    )
    # Apply the ColumnTransformer to the data
    data_features = column_transformer.fit_transform(data)
    # Combine the transformed features with the target column
    data_transformed = pd.DataFrame(data_features, columns=columns)


    return data_transformed

# Function to impute missing values and use convert_to_numerical function
def impute_missing_values(data, strategy='mean'):
    # Convert categorical and object columns to numerical
    data = convert_to_numerical(data)
    # Create a SimpleImputer object
    imputer = SimpleImputer(strategy=strategy)
    # Apply the imputer to the data
    data = imputer.fit_transform(data)
    return data

# Function to impute missing values using KNNImputer
def impute_missing_values_knn(data, n_neighbors=5):
    # Convert categorical and object columns to numerical
    data = convert_to_numerical(data)
    # Create a KNNImputer object
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # Apply the imputer to the data
    data = imputer.fit_transform(data)
    return data

# function to remove one or more columns
def remove_columns(data, columns):
    # Remove the columns
    data = data.drop(columns=columns)
    return data

def handle_missing_values(data, dataset_name):
            if dataset_name == 'Titanic':
                # Section to handle missing values separately
                st.write('## Handle Missing Values')

            # Option to drop columns with missing values
def drop_columns_with_missing_values(data):
    st.write('### Drop Columns with Missing Values')
    if st.checkbox('Select to drop columns'):
        columns_to_drop = st.multiselect('Select columns to drop', data.columns[data.isnull().any()])
        if columns_to_drop:  # Perform the drop only if the user selected some columns
            data = remove_columns(data, columns_to_drop)
            st.write('Columns dropped:', columns_to_drop)
    return data 

def select_imputation_strategy(data ):
    # Section for imputation options
    st.write('### Impute Missing Values')
    impute_options = st.selectbox('Choose an imputation method', 
                                    ['None',  # Add 'None' option for no imputation
                                    'Impute with mean (for numerical columns)', 
                                    'Impute with median (for numerical columns)',
                                    'Impute with mode (for categorical columns)',
                                    'Impute with KNN'])


    # Conditional logic based on the choice of imputation
    if 'Impute with' in impute_options:
        data = apply_imputation(data, impute_options)
    return data 

def display_data_after_missing_values(data):
    st.write('### Data after Handling Missing Values')
    st.write(pd.DataFrame(data).head())


def apply_imputation(data, impute_options):
    
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


# create function for scaling data
def scale_data(data, key):
    st.write('## Data Scaling')
    is_scaled = False
    # Ask user if they want to scale data
    scale_option = st.selectbox('Do you want to scale the data?', ('No', 'Yes'), key=key)

    if scale_option == 'Yes':     
        st.write('## Data after Scaling')
          # Create a StandardScaler object
        scaler = StandardScaler()
        # Apply the scaler to the data
        data = scaler.fit_transform(data)
        st.write(pd.DataFrame(data).head())
        is_scaled = True

    else:
        st.write('## Data without Scaling')
        st.write(pd.DataFrame(data).head())
        is_scaled = False
   

    return data, is_scaled
def initialize_session_state():
    if 'target_variable' not in st.session_state:
        st.session_state['target_variable'] = None

initialize_session_state()





def encode_target(data, target_column):
    if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
    return data, label_encoder

def inverse_encode(data, target_column):
    if st.session_state['is_encoded']:
        data[target_column] = st.session_state['label_encoder'].inverse_transform(data[target_column])
    return target_column

if 'is_encoded' not in st.session_state:
    st.session_state['is_encoded'] = False

# Initialize session state
if 'label_encoder' not in st.session_state:
    st.session_state['label_encoder'] = None

def handle_target_variable(data, target_column):
    if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
        st.write('### Target Variable Value Counts')
        st.write(data[target_column].value_counts())              
          
        data, le = encode_target(data, target_column)
        st.session_state['is_encoded'] = True
        st.session_state['label_encoder'] = le

        # st.write('Target variable encoded')
        # unique_values = pd.DataFrame(data[target_column].unique(), columns=[target_column])
        # st.write(unique_values)
          
    return data

# initialise state_session with the target column
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None

# Streamlit app
def main():
    
    dataset_name = display_dataset()
    data = load_data(dataset_name)
    
    columns = [column for column in data.columns]

    st.write('### Select the Target Variable')
    target_column = st.selectbox('Select the target column', data.columns)
    target_data = data[target_column]

    data = handle_target_variable(data, target_column)    
    data = pd.DataFrame(data, columns=columns)

    target_column_data = data[target_column]
    data.drop(target_column, axis=1, inplace=True)
    cols_without_target = [col for col in data.columns]
 
   
    
    # check if the data is not None, load and show data
    if data is not None:

        # concat at axis =1 data and target column data into a new variable
        read_only_data = pd.concat([data, target_data], axis=1)
        display_data_overview(read_only_data)
        display_categorical_columns(read_only_data)
        display_most_frequent_values(read_only_data)
        display_unique_values(read_only_data)
        display_missing_values(read_only_data, dataset_name)
       
        
        if data.isnull().sum().any() > 0:
           
            st.markdown('---')
            handle_missing_values(data, dataset_name)
            data = drop_columns_with_missing_values(data)
            cols_without_target = data.columns

            data = select_imputation_strategy(data)
            display_data_after_missing_values(data)
            
            
        # check if there are any catergorical data before encoding
        if pd.DataFrame(data).select_dtypes(include=['object', 'category']).shape[1] > 0:        
            data = convert_to_numerical(data)
        st.markdown('---')
     
  

        # Scaling data
        features_data_scaled, is_scaled = scale_data(data, key='sc_01') 
        data = pd.DataFrame(features_data_scaled, columns=cols_without_target)
       
        
        #columns.remove(target_column)
        data[target_column] = target_column_data
           
       
        
        # give the target column data
        if is_scaled:   
            # dumb the data to a csv file in data folder which same as parent directory hierarchy use pandas library
            data.to_csv(f'data/data_{dataset_name}.csv', index=False)


if __name__ == "__main__":
    main()
