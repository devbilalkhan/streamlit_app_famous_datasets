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

from sklearn.preprocessing import StandardScaler

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
    st.write(data.head())
    st.write('## Data Description')
    st.write(data.describe())

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

def select_imputation_strategy(data):
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

def split_data(data, target_column):
        
        test_size = st.slider(
            'Select the test set percentage',
            min_value=5,
            max_value=40,
            value=20,  # Default test set percentage
            step=10,
        )

        train_size = 100 - test_size
        st.write(f"Training set percentage: {train_size}%")
        st.write(f"Testing set percentage: {test_size}%")
    

        # Split the data into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        return X_train, X_test, y_train, y_test

def model_preprocess(data, columns, target_column):
            # Define the slider for selecting the split percentage
        # heading of Model Train
        st.write('## Model Training')
        st.write('### Split Dataset Ratio')
            
        # Split the data into features and target
        # features_columns = st.multiselect('Select the features columns', columns)
        
        #convert the data into pandas format and add columns
        data = pd.DataFrame(data, columns=columns)
        
        
        
        X_train, X_test, y_train, y_test = split_data(data, target_column)
        st.write('### Data after Splitting')
        st.write('#### Training Set')
        st.write(X_train.head())
        # add one line text the number of rows
        st.write('Number of rows in the training set:', X_train.shape[0])


        st.write('#### Testing Set')
        st.write(X_test.head())
        # add one line text the number of rows
        st.write('Number of rows in the testing set:', X_test.shape[0])

        return X_train, X_test, y_train, y_test

# create a function for Random forest and give streamlit hyperparameters

def random_forest(X_train, X_test, y_train, y_test):
    st.write('## Random Forest')
    from sklearn.ensemble import RandomForestClassifier
    # Create a RandomForestClassifier object
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)


# Streamlit app
def main():
    st.title('@Models Testing')
    st.markdown('---')
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Diamonds', 'Tips', 'Titanic'))  # Add your other dataset to the list

    data = load_data(dataset_name)
    
    columns = [column for column in data.columns]

    st.write('### Select the Target Variable')
    target_column = st.selectbox('Select the target column', columns)
    

    # check if the data is not None, load and show data
    if data is not None:
        display_data_overview(data)
     
        display_categorical_columns(data)
        display_most_frequent_values(data)
        display_unique_values(data)
        display_missing_values(data, dataset_name)
        
        if data.isnull().sum().any() > 0:
            st.markdown('---')
            handle_missing_values(data, dataset_name)
            data = drop_columns_with_missing_values(data)
            data = select_imputation_strategy(data)
            display_data_after_missing_values(data)
        
            

        # check if there are any catergorical data before encoding
        if pd.DataFrame(data).select_dtypes(include=['object', 'category']).shape[1] > 0:        
            data = convert_to_numerical(data)
        st.markdown('---')
        #scale the data
        data, is_scaled = scale_data(data, key='sc_01')

        if is_scaled:
            st.markdown('---')

            # Ask user if they want to display the section
            display_option = st.selectbox('Do you want train the model?', ('No', 'Yes'))

            if display_option == 'Yes':         
                X_train, X_test, y_train, y_test = model_preprocess(data, columns, target_column)

if __name__ == "__main__":
    main()
