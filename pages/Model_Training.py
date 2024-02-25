# Model_Training.py
import streamlit as st
import plotly.express as px   
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from utils import set_color_map
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from models.default_models import models_dict
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from db.client import get_database
from db.crud import insert_documents
from config import DATABASE_NAME, MODEL_RESULTS_COLLECTION
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from utils import get_model_params, load_data, clean_dataset_name, select_existing_datasets


# create a state session called is_selected  with the condition whether it exists or not
if 'is_selected' not in st.session_state:
    st.session_state.is_selected = False

def split_data(data, target_column):
    """
    Splits the data into training and testing sets based on a user-selected test set percentage.

    Parameters:
    data (pandas.DataFrame): The DataFrame to split.
    target_column (str): The target column.

    Returns:
    pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series: The training features, testing features, training target, and testing target.
    """
    test_size = st.slider(
        'Select the test set percentage',
        min_value=5,
        max_value=40,
        value=20,  # Default test set percentage
        step=5,
    )

    train_size = 100 - test_size
    st.write(f"Training set percentage: {train_size}%")
    st.write(f"Testing set percentage: {test_size}%")

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    return X_train, X_test, y_train, y_test


def model_preprocess(data, columns, target_column):
    """
    Preprocesses the data for model training, including splitting the data into training and testing sets.

    Parameters:
    data (pandas.DataFrame): The DataFrame to preprocess.
    columns (list): The list of columns in the DataFrame.
    target_column (str): The target column.

    Returns:
    pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series: The training features, testing features, training target, and testing target.
    """
    st.write('## Model Training')
    st.write('### Split Dataset Ratio')

    data = pd.DataFrame(data, columns=columns)

    X_train, X_test, y_train, y_test = split_data(data, target_column)
    st.write('### Data after Splitting')
    st.write('#### Training Set')
    st.write(X_train.head())
    st.write('Number of rows in the training set:', X_train.shape[0])

    st.write('#### Testing Set')
    st.write(X_test.head())
    st.write('Number of rows in the testing set:', X_test.shape[0])

    return X_train, X_test, y_train, y_test


def model_evaluation_regression(results, model_name, y_test, y_pred_test):
    """
    Evaluates a regression model by calculating various metrics.

    Parameters:
    results (dict): The dictionary to store the results in.
    model_name (str): The name of the model.
    y_test (pandas.Series): The actual target values for the test set.
    y_pred_test (pandas.Series): The predicted target values for the test set.

    Returns:
    dict: The results dictionary updated with the calculated metrics.
    """
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    results[model_name] = {
        'Test MSE': test_mse,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test R2': test_r2,
    }
    return results


def model_evaluation_classification(results, model_name, y_test, y_pred_test):
    """
    Evaluates a classification model by calculating various metrics.

    Parameters:
    results (dict): The dictionary to store the results in.
    model_name (str): The name of the model.
    y_test (pandas.Series): The actual target values for the test set.
    y_pred_test (pandas.Series): The predicted target values for the test set.

    Returns:
    dict: The results dictionary updated with the calculated metrics.
    """
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    test_f1_score = f1_score(y_test, y_pred_test, average='weighted')

    results[model_name] = {
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1 Score': test_f1_score,
    }
    return results


def select_problem_type_and_models(models_dict):
    """
    Allows the user to select the problem type and models to run.

    Parameters:
    models_dict (dict): The dictionary of models.

    Returns:
    str, list, str, list: The selected problem type, selected models, select all option, and model options.
    """
    st.markdown('---')
    st.write('### Model Training Options')
    problem_type = st.radio('Select the problem type:', ('Classification', 'Regression'))
    select_all_option = "Select All"
    model_options = list(models_dict[problem_type].keys())
    options = [select_all_option] + model_options
    selected_models = st.multiselect(f'Select the models to run for {problem_type}:', options)
    return problem_type, selected_models, select_all_option, model_options


def handle_select_all(selected_models, select_all_option, model_options):
    """
    Handles the select all option for model selection.

    Parameters:
    selected_models (list): The list of selected models.
    select_all_option (str): The select all option.
    model_options (list): The list of model options.

    Returns:
    list: The updated list of selected models.
    """
    if select_all_option in selected_models:
        if len(selected_models) == 1 or set(selected_models) == set(model_options + [select_all_option]):
            selected_models = model_options
        else:
            selected_models.remove(select_all_option)
    return selected_models


def display_model_params(selected_models, models_dict, problem_type):
    """
    Displays the parameters for the selected model.

    Parameters:
    selected_models (list): The list of selected models.
    models_dict (dict): The dictionary of models.
    problem_type (str): The selected problem type.
    """
    if len(selected_models) == 1:
        st.session_state.is_selected = True
        if selected_models[0] in ['CatBoost', 'XGBoost', 'LightGBM']:
            params = get_model_params(selected_models[0])
            st.write(f'You selected {selected_models[0]}. You can customize the hyperparameters below:')
            st.write(params)
            model_name = selected_models[0]
            models_dict[problem_type][model_name].set_params(**params)
        
        if selected_models[0] in ['RandomForest', 'DecisionTree', 'AdaBoost', 'GradientBoosting']:
            params = get_model_params(selected_models[0])
            st.write(f'You selected {selected_models[0]}. You can customize the hyperparameters below:')
            st.write(params)
            model_name = selected_models[0]
            models_dict[problem_type][model_name].set_params(**params)


def train_and_evaluate_models(problem_type, selected_models, models_dict, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the selected models.

    Parameters:
    problem_type (str): The selected problem type.
    selected_models (list): The list of selected models.
    models_dict (dict): The dictionary of models.
    X_train (pandas.DataFrame): The training features.
    X_test (pandas.DataFrame): The testing features.
    y_train (pandas.Series): The training target.
    y_test (pandas.Series): The testing target.

    Returns:
    dict, dict: The results and hyperparameters.
    """
    results = {}
    hyperparameters = {}
    for model_name in selected_models:
        with st.spinner(f'Training {model_name}...'):
            model = models_dict[problem_type][model_name]
            model.fit(X_train, y_train)

            hyperparameters[model_name] = model.get_params()

            y_pred_test = model.predict(X_test)
            if problem_type == 'Classification':
                y_pred_test_proba = model.predict_proba(X_test)
                results = model_evaluation_classification(results, model_name, y_test, y_pred_test)
            elif problem_type == 'Regression':
                results = model_evaluation_regression(results, model_name, y_test, y_pred_test)
    return results, hyperparameters


def save_and_display_results(results, dataset_name, hyperparameters, problem_type):
    """
    Saves and displays the results of model training.

    Parameters:
    results (dict): The dictionary of results.
    dataset_name (str): The name of the dataset.
    hyperparameters (dict): The dictionary of hyperparameters.
    problem_type (str): The type of problem (classification or regression).

    Returns:
    None
    """
    if results:
        results_df = pd.DataFrame(results).T
        hyperparameters_df = pd.DataFrame(hyperparameters).T
        st.table(results_df)

        filtered_hyperparameters = {
            model: {param: value for param, value in params.items() if value is not None}
            for model, params in hyperparameters.items()
        }
        hyperparameters_df = pd.DataFrame(filtered_hyperparameters).T

        results_with_model_column = results_df.reset_index()
        results_with_model_column.rename(columns={'index': 'Model'}, inplace=True)
        results_with_model_column['DatasetName'] = dataset_name
        results_with_model_column['ProblemType'] = problem_type

        combined_results = results_with_model_column.join(hyperparameters_df, on='Model')

        db = get_database('DATABASE_NAME')
        
        insert_documents(combined_results, MODEL_RESULTS_COLLECTION)
        st.success('Models trained and results saved successfully! Check the Dashboard for the results.')


def train_fit_models(data, columns, target_column, dataset_name, models_dict):
    """
    Trains and fits models based on the provided data.

    Parameters:
    data (pandas.DataFrame): The DataFrame to train the models on.
    columns (list): The list of columns in the DataFrame.
    target_column (str): The target column.
    dataset_name (str): The name of the dataset.
    models_dict (dict): The dictionary of models.

    Returns:
    None
    """
    if data is not None:
        X_train, X_test, y_train, y_test = model_preprocess(data, columns, target_column)
        problem_type, selected_models, select_all_option, model_options = select_problem_type_and_models(models_dict)
        selected_models = handle_select_all(selected_models, select_all_option, model_options)
        display_model_params(selected_models, models_dict, problem_type)

        if st.button('Train Selected Models'):
            st.session_state.is_selected = False
            results, hyperparamters = train_and_evaluate_models(problem_type, selected_models, models_dict, X_train, X_test, y_train, y_test)
            save_and_display_results(results, dataset_name, hyperparamters, problem_type)


# Main
def main():
    # Set the title of the page
    st.title('🤖 Model Training')
    # Display a message
    st.write('This is the Model Training page. Feed me data!!')
    dataset_name = select_existing_datasets('dataset_names')
    dataset_name = clean_dataset_name(dataset_name)   

    # strip the datanames from any whitespace and hyphens and replave with underscores       
    dataset_name = dataset_name.lower()
    data = pd.read_csv(f'data/{dataset_name}.csv')

   
    columns = [column for column in data.columns]
    # Display a message
    st.write('### Select the Target Variable')
    # Allow the user to select the target column
    target_column = st.selectbox('Select the target column', data.columns)
    # Train and fit models based on the selected dataset and target column
    train_fit_models(data, columns, target_column, dataset_name, models_dict)
    


if __name__ == '__main__':
    main()