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
    
    if target_column and target_column in cat_columns:
        cat_columns.remove(target_column)
    
    for column in cat_columns:
        data[column] = data[column].astype(str)
    
    # Create column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), cat_columns),
        ],
        remainder='passthrough'
    )
    
    data_features = column_transformer.fit_transform(data)
    
    new_columns = cat_columns + [col for col in data.columns if col not in cat_columns]
    data_transformed = pd.DataFrame(data_features, columns=new_columns, index=data.index)
    
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
    data = data.drop(columns=columns, axis=1)
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
    st.write('#### Drop Columns with Missing Values')
    if st.checkbox('Select to drop columns with missing values'):
        columns_to_drop = st.multiselect('Select columns to drop', data.columns[data.isnull().any()])
        if st.button('Drop Columns'):
            if columns_to_drop:
                data = remove_columns(data, columns_to_drop)
                st.write('Columns dropped:', columns_to_drop)
    return data 

def drop_columns(data):
    """
    Drops selected columns from the data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to drop columns from.

    Returns:
    pandas.DataFrame: The DataFrame with selected columns dropped.
    """
    st.write('### Drop Columns')
    columns_to_drop = st.multiselect('Select columns to drop', data.columns)
    if st.button('Drop Columns'):
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
    st.write('#### Impute Missing Values')
    impute_options = st.selectbox('Choose an imputation method', 
                                    ['None', 
                                    'Impute with Simple Imputer', 
                                    ])

    if 'Impute with' in impute_options:
        data = apply_simple_imputation(data)
    return data 

def display_data_after_missing_values(data):
    """
    Displays the data after handling missing values.

    Parameters:
    data (pandas.DataFrame): The DataFrame to display.
    """
    st.write('#### Data after Handling Missing Values')
    st.write(pd.DataFrame(data).head())

def apply_simple_imputation(data):
    """
    Applies mean imputation to numerical data and most frequent imputation to categorical data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to impute missing values in.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    # Imputers
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numerical and categorical columns
    numerical_data = data.select_dtypes(include=['number'])
    categorical_data = data.select_dtypes(exclude=['number'])

    # Apply imputation
    imputed_numerical_data = num_imputer.fit_transform(numerical_data)
    imputed_categorical_data = cat_imputer.fit_transform(categorical_data)

    # Convert imputed arrays back to DataFrames
    imputed_numerical_data = pd.DataFrame(imputed_numerical_data, columns=numerical_data.columns, index=numerical_data.index)
    imputed_categorical_data = pd.DataFrame(imputed_categorical_data, columns=categorical_data.columns, index=categorical_data.index)

    # Concatenate the imputed numerical and categorical data
    data_imputed = pd.concat([imputed_numerical_data, imputed_categorical_data], axis=1)

    # Ensure the original order of columns is preserved
    data_imputed = data_imputed.reindex(data.columns, axis=1)

    # display the data after imputation
    display_data_after_missing_values(data_imputed)
    return data_imputed


def scale_data(data, scale_option):
    """
    Scales the data if the user selects to do so.

    Parameters:
    data (pandas.DataFrame): The DataFrame to scale.
    key (str): The key for the Streamlit widget.

    Returns:
    pandas.DataFrame, bool: The potentially scaled DataFrame and a flag indicating if scaling was applied.
    """
   
    is_scaled = False
    # check if there are missing values in the data if you give me message to user to remove it
    if data.isnull().sum().sum() > 0:        
        st.write('Note: Please handle missing values before scaling the data.')
        return data, is_scaled

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


def handle_file_upload():
    """
    Handles the file upload process and returns the uploaded data and the dataset name.

    Returns:
    tuple or None: (pandas.DataFrame, str) with the uploaded data and dataset name, or None if no file was uploaded.
    """
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
   
    if uploaded_file is not None:
        dataset_name = st.sidebar.text_input("Enter a name for the dataset (max 3 words):", key='dataset_name')
        word_count = len(dataset_name.split())
        if word_count > 3:
            st.sidebar.write("Please enter a name that is no more than three words.")
            return None
        elif word_count == 0:
            st.sidebar.error("Please enter a name for the dataset.")
            return None
        else:         
            data = pd.read_csv(uploaded_file, nrows=ROWS)
            dataset_name = check_and_create_single_ds_name(dataset_name.lower(), DATASET_COLLECTION_NAME)
            st.write(f"Dataset {dataset_name} loaded successfully!")
            st.write(data.head())
            return data, dataset_name
    return None


def delete_ds(collection, dataset_name):
    """
    Deletes a dataset from the specified collection.

    Parameters:
    collection (str): The name of the collection to delete the dataset from.
    """
    
    delete_dataset_records(collection, dataset_name)
    st.write(f"Dataset {dataset_name} deleted successfully!")

def remove_outliers(data, method='IQR'):
    """
    Removes outliers from the numerical columns of a DataFrame using the specified method.
    Parameters:
    - data (pandas.DataFrame): The DataFrame from which to remove outliers.
    - method (str): The method to use for outlier removal ('Standard Deviation' or 'IQR').
    Returns:
    - pandas.DataFrame: A new DataFrame with outliers removed from numerical columns.
    """
    # Allow the user to select the method for outlier removal
    method = st.selectbox('Select a method for outlier removal:', ['Standard Deviation', 'Interquartile Range (IQR)'])
    # Select only numerical columns
    plot_qq_for_selected_feature(data)
   
    numerical_data = data.select_dtypes(include=['number'])

    if st.button('Remove Outliers'):
        if method == 'Standard Deviation':
            # Assuming normally distributed data, we'll consider data points within 3 standard deviations
            mean = numerical_data.mean()
            std_dev = numerical_data.std()
            data_no_outliers = numerical_data[(np.abs(numerical_data - mean) <= (3 * std_dev)).all(axis=1)]

        elif method == 'Interquartile Range (IQR)':
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
            Q1 = numerical_data.quantile(0.25)
            Q3 = numerical_data.quantile(0.75)
            IQR = Q3 - Q1

            # Define upper and lower bounds for what you consider to be an outlier
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            data_no_outliers = numerical_data[~((numerical_data < lower_bound) | (numerical_data > upper_bound)).any(axis=1)]
        
        else:
            raise ValueError("Unknown method selected! Use 'Standard Deviation' or 'IQR'")

        # Combine the numerical and non-numerical data back into one DataFrame
        non_numerical_data = data.select_dtypes(exclude=['number']).loc[data_no_outliers.index]
        data_cleaned = pd.concat([data_no_outliers, non_numerical_data], axis=1)
        num_rows_removed = len(data) - len(data_no_outliers)
        
        # Display the number of rows that were considered outliers
        st.write(f'Number of rows considered outliers: {num_rows_removed}')
        
        return data_cleaned

def plot_missing_values(data):
    """
    Plots a heatmap of missing values in the DataFrame.
    
    Parameters:
    - data (pandas.DataFrame): The DataFrame for which to plot missing values.
    """
    from matplotlib.colors import ListedColormap
    # Check if there's any missing values in the DataFrame
    if data.isnull().sum().sum() == 0:
        st.write('No missing values found in the dataset!')
        return

    # When the button is clicked, plot the missing values heatmap
    if st.button('Show Missing Values Plot'):
        # Create a custom colormap
        # (0, 0, 0, 0) is transparent (RGBA) for non-missing (False) values
        # 'indigo' or the equivalent RGBA tuple for missing (True) values
        cmap = ListedColormap([(0, 0, 0, 0), 'indigo'])

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap=cmap)
        plt.title('Missing Values Heatmap', color='white')  # Set title color

        # Set the color of the xticklabels and yticklabels to white or light gray
        plt.xticks(color='white')
        plt.yticks(color='white')

        # Set the background to transparent
        plt.gca().patch.set_alpha(0)
        plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

        # Show the plot in Streamlit
        st.image('heatmap.png')



def plot_qq_for_selected_feature(data):
    """
    Allows the user to select a feature from the DataFrame and plots a QQ plot for it.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the features.
    """
    import scipy.stats as stats
    import plotly.graph_objs as go
    st.write('##### QQ Plot for Selected Feature')
    shades = ['#9468F8', '#8B30E3', '#7039FF', '#341F9B', '#9556EB']
    # Select a feature to plot

    
    feature = st.selectbox('Select a feature to plot QQ plot:', data.columns)

    # When the button is clicked, plot the QQ plot for the specified feature
    if st.button(f'Plot QQ for {feature}'):
        # Check if feature contains numeric data
        if pd.api.types.is_numeric_dtype(data[feature]):
            st.write(f'QQ plot for {feature}:')
            
            # Calculate quantiles and least-square-fit line
            data_array = data[feature].dropna()
            qq = stats.probplot(data_array, dist="norm")
            x = np.array([qq[0][0][0], qq[0][0][-1]])  # Theoretical quantiles

            # Create Plotly graph
            qq_trace = go.Scatter(
                x=qq[0][0], 
                y=qq[0][1], 
                mode='markers', 
                name='Data', 
                marker=dict(
                    color=shades[0]
                )
            )
            qq_line = go.Scatter(
                x=x, 
                y=qq[1][1] + qq[1][0]*x, 
                mode='lines', 
                name='Fit', 
                marker=dict(
                    color=shades[2]
                )
            )

            fig = go.Figure(data=[qq_trace, qq_line])
            fig.update_layout(
                title=f'Quantile-Quantile Plot of {feature}',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f'The selected feature "{feature}" is not numeric and cannot be plotted in a QQ plot.')



# Main
def main():
 
    st.sidebar.title("Dataset Options")
    st.title("Data Cleaning ⚙️")
    st.write('This your playground to clean your dataset. You can select an existing dataset or upload a new one. ')
    placeholder_success = st.empty()
    # Ask the user what they want to do
    dataset_option = st.sidebar.radio("Choose an option",
                                      ('Select Existing Dataset', 'Upload New Dataset'))
    data = None
    if dataset_option == 'Select Existing Dataset':
        dataset_name = select_existing_datasets('dataset_names')
        data = load_data(dataset_name)
        
        # if data is none then load the data from data folder csv file
        if data is None:
            # strip the datanames from any whitespace and hyphens and replave with underscores
            dataset_name = clean_dataset_name(dataset_name)
            dataset_name = dataset_name.lower()
            data = pd.read_csv(f'data/{dataset_name}.csv')

        st.write(f"Loaded dataset: {dataset_name}")
        st.write(data.head())
        # Process the data from the file path or database reference here

    elif dataset_option == 'Upload New Dataset':
        result = handle_file_upload()      
        if result is not None:
            data, dataset_name = result           
            # strip the datanames from any whitespace and hyphens and replave with underscores
            dataset_name = clean_dataset_name(dataset_name)
            # Save to csv
            data.to_csv(f'data/{dataset_name.lower()}.csv', index=False)
            placeholder_success.success(f"Dataset {dataset_name} saved successfully!") 

    if data is not None:
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
       
        read_only_data = pd.concat([data, target_data], axis=1)
        display_data_overview(read_only_data)
        display_categorical_columns(read_only_data)
        display_most_frequent_values(read_only_data)
        display_unique_values(read_only_data)
        display_missing_values(read_only_data, dataset_name)

        st.markdown('---')
        data = drop_columns(read_only_data)
        cols_without_target = data.columns

        # If there are missing values, handle them
        if data.isnull().sum().any() > 0:
            st.markdown('---')
            #handle_missing_values(data, dataset_name)
            st.write('## Missing Values')
            plot_missing_values(data)
            
            data = select_imputation_strategy(data)           
            st.markdown('---')
            data = pd.DataFrame(data, columns=cols_without_target)
        
        # If there are categorical columns, convert them to numerical
        if pd.DataFrame(data).select_dtypes(include=['object', 'category']).shape[1] > 0:        
            data = convert_to_numerical(data)

        # Scale the data if 
        st.write('## Scale Data')
        scale_option = st.selectbox('Do you want to scale the data?', ('No', 'Yes'), key='sc_01')
        features_data_scaled = None
        is_scaled = False
        if scale_option == 'Yes':
            features_data_scaled, is_scaled = scale_data(data, scale_option) 
            data = pd.DataFrame(features_data_scaled, columns=cols_without_target)
        # st.write('## Outlier Removal')
        # if st.selectbox('Remove Outliers', ['No', 'Yes'], key="scale_data") == 'Yes': 
        #     data = remove_outliers(data)
        
        # Add the target column back to the data
            data[target_column] = target_column_data
        #If the data was scaled, save it to a CSV file
        
            data.to_csv(f'data/{dataset_name.lower()}.csv', index=False)
    else:
       display_welcome_message()           
    placeholder_success.empty()


if __name__ == "__main__":
    main()
