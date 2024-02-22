import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from utils import set_color_map
from sklearn.model_selection import train_test_split

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

# Function to split data into train and test sets
def split_data(data, test_size=0.2):

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data


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
def convert_to_numerical(data):
    # Convert categorical and object columns to numerical
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    cat_columns = list(cat_columns)
    # Create a ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), cat_columns)
        ],
        remainder='passthrough'
    )
    # Apply the ColumnTransformer to the data
    data = column_transformer.fit_transform(data)
    return data

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



# Streamlit app
def main():
    st.title('@Models Testing')
    
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Diamonds', 'Tips', 'Titanic'))  # Add your other dataset to the list

    data = load_data(dataset_name)
    train_ds, test_ds = split_data(data)

    # check if the data is not None, load and show data
    if data is not None:
        st.write('## Data')
        st.write(data.head())
    
        st.write('## Data Description')
        st.write(data.describe())
        #display categorical columns only
        cat_columns = data.select_dtypes(include=['object', 'category']).columns

        st.write('## Categorical Columns')
        # Convert Index object to DataFrame for pretty display
        cat_columns_df = pd.DataFrame(cat_columns, columns=['Categorical Features'])
        st.write(cat_columns_df)

        # most frequent values
        st.write('## Most Frequent Values')
        most_frequent_values(data)

        # most unique values
        st.write('## Unique Values')
        unique_values(data)

        # Call the function with your DataFrame
        display_missing_values(data, name='Titanic')
        if dataset_name == 'Titanic':
            # Section to handle missing values separately
            st.write('## Handle Missing Values')

            # Option to drop columns with missing values
            st.write('### Drop Columns with Missing Values')
            if st.checkbox('Select to drop columns'):
                columns_to_drop = st.multiselect('Select columns to drop', data.columns[data.isnull().any()])
                if columns_to_drop:  # Perform the drop only if the user selected some columns
                    data = remove_columns(data, columns_to_drop)
                    st.write('Columns dropped:', columns_to_drop)

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

            st.write('## Data after Handling Missing Values')
            st.write(pd.DataFrame(data).head())



    

if __name__ == "__main__":
    main()
