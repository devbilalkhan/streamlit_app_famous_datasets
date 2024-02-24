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

from db.client import get_database
from db.crud import insert_documents
from config import DATABASE_NAME, MODEL_RESULTS_COLLECTION

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

from pymongo import MongoClient




def split_data(data, target_column):
        
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
def model_evaluation_regression(results, model_name, y_test, y_pred_test):
    
    
   
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

def model_evaluation_classification(results, model_name, y_pred_test_proba, y_test, y_pred_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
   
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
   
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
   
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    # train_auc = roc_auc_score(y_train, y_pred_train_proba, multi_class='ovr')
    # test_auc = roc_auc_score(y_test, y_pred_test_proba, multi_class='ovr')

    results[model_name] = {
       
        'Test Accuracy': test_accuracy,
       
        'Test Precision': test_precision,
       
        'Test Recall': test_recall,
       
        'Test F1': test_f1,
        # 'Train AUC': train_auc,
 
    }

    return results


def train_fit_models(data, columns, target_column, dataset_name):
 

    # Assuming 'data' is your DataFrame and 'features' and 'labels' are your feature matrix and labels
    if data is not None:
        # Split your data
        X_train, X_test, y_train, y_test = model_preprocess(data, columns, target_column)
        
        # Let the user select the problem type
        problem_type = st.radio('Select the problem type:', ('Classification', 'Regression'))
        # Define your 'Select All' option

        select_all_option = "Select All"

        # Get the list of model names for the problem type
        model_options = list(models_dict[problem_type].keys())

        # Add 'Select All' option at the start of the model options list
        options = [select_all_option] + model_options

        # Use a multiselect widget to let the user select models
        selected_models = st.multiselect(f'Select the models to run for {problem_type}:', options)

        # If user selects 'Select All', update selected_models to include all models
        if select_all_option in selected_models:
            # If only 'Select All' is selected or it's selected with all other models, select all models
            if len(selected_models) == 1 or set(selected_models) == set(options):
                selected_models = model_options
            else:
                # If 'Select All' is selected with a subset of models, remove 'Select All' from the list
                selected_models.remove(select_all_option)

        # Now, you can safely use selected_models to access your models_dict
        for model_name in selected_models:
            model = models_dict[problem_type][model_name]
            # Your code to train and fit the model goes here

        # # (Optional) To reflect the selection in the UI after 'Select All' logic
        # st.multiselect(f'Select the models to run for {problem_type}:', options, default=selected_models)
       
        # Add a button to start the training process
        if st.button('Train Selected Models'):
            # Initialize an empty dictionary to store results
            results = {}

            # Loop through the selected models and run them
            for model_name in selected_models:
                with st.spinner(f'Training {model_name}...'):
                    # Initialize the model
                    model = models_dict[problem_type][model_name]
                    # Train the model
                    model.fit(X_train, y_train)
                    # Make predictions
                    y_pred_test = model.predict(X_test)

                    # Calculate probabilities for classification
                    if problem_type == 'Classification':
                        y_pred_test_proba = model.predict_proba(X_test)

                    # Evaluate models based on problem type
                    if problem_type == 'Regression':
                        results = model_evaluation_regression(results, model_name, y_test, y_pred_test)
                    if problem_type == 'Classification':
                        results = model_evaluation_classification(results, model_name, y_pred_test_proba, y_test, y_pred_test)

            # Process results after training
            if results:
                results_df = pd.DataFrame(results).T
                st.table(results_df)

                # Reset the index to turn the model names from the index into a column
                results_with_model_column = results_df.reset_index()

                # Rename the 'index' column to 'Model'
                results_with_model_column.rename(columns={'index': f'{problem_type}'}, inplace=True)
                 # Add the dataset name to a new column
                results_with_model_column['DatasetName'] = dataset_name
                
                db = get_database(DATABASE_NAME)
                collection = db[MODEL_RESULTS_COLLECTION]

                # Insert records into the database
                insert_documents(results_with_model_column, MODEL_RESULTS_COLLECTION)

               

                # Notify the user of success
                st.success('Models trained and results saved successfully! Check the Dashboard for the results.')


def main():
    st.title('🤖 Model Training')
    st.write('This is the Model Training page')
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Diamonds', 'Tips', 'Titanic'))
    # Load the data
   # Check if the file exists
    # if os.path.isfile(f'data/data_{dataset_name}.csv'):
        # Load the data
    data = pd.read_csv(f'data/data_{dataset_name}.csv')
    columns = [column for column in data.columns]
    st.write('### Select the Target Variable')
    target_column = st.selectbox('Select the target column', data.columns)
    train_fit_models(data, columns, target_column, dataset_name)
    # else:
    #     st.write("#### The data is not ready to be fed by robots. Please clean the data first!")
   


if __name__ == '__main__':
    main()