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
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


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
def model_evaluation_regression(results, model_name, y_train, y_pred_train, y_test, y_pred_test):
    
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = sqrt(train_mse)
    test_rmse = sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("=====>", train_r2)

    results[model_name] = {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'Test R2': test_r2,
    }
    return results

def model_evaluation_classification(results, model_name, y_train, y_pred_train, y_test, y_pred_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    train_recall = recall_score(y_train, y_pred_train, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    # train_auc = roc_auc_score(y_train, y_pred_train_proba, multi_class='ovr')
    # test_auc = roc_auc_score(y_test, y_pred_test_proba, multi_class='ovr')

    results[model_name] = {
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Train Precision': train_precision,
        'Test Precision': test_precision,
        'Train Recall': train_recall,
        'Test Recall': test_recall,
        'Train F1': train_f1,
        'Test F1': test_f1,
        # 'Train AUC': train_auc,
        # 'Test AUC': test_auc,
    }

    return results


def train_fit_models(data, columns, target_column):
    
  
    # ... (import other necessary libraries and models)

    # Assuming 'data' is your DataFrame and 'features' and 'labels' are your feature matrix and labels
    if data is not None:
       

        # Split your data
        X_train, X_test, y_train, y_test = model_preprocess(data, columns, target_column)
         # Let the user select the problem type
        problem_type = st.radio('Select the problem type:', ('Classification', 'Regression'))

        # Define the model options
        models_dict = {
            'Classification': {
                'Logistic Regression': LogisticRegression(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Machine': SVC(),               
                'RandomForest': RandomForestClassifier(),
                'CatBoost': CatBoostClassifier(),
                'XGBoost': XGBClassifier(),
                'LightGBM': LGBMClassifier(),
            },
            'Regression': {
                'Linear Regression': LinearRegression(),
                'Support Vector Machine': SVR(),
                'RandomForest': RandomForestRegressor(),
                'CatBoost': CatBoostRegressor(),
                'XGBoost': XGBRegressor(),
                'LightGBM': LGBMRegressor(),            
           
            }
        }

        # Let the user select multiple models
        selected_models = st.multiselect(f'Select the models to run for {problem_type}:', list(models_dict[problem_type].keys()))
        # Initialize an empty dictionary to store results
        results = {}

        # Loop through the selected models and run them
        for model_name in selected_models:
            with st.spinner(f'Training model...'):
                # Initialize the model
                model = models_dict[problem_type][model_name]
                # Train the model
                model.fit(X_train, y_train)
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                if problem_type == 'Regression':
                    results = model_evaluation_regression(results, model_name, y_train, y_pred_train, y_test, y_pred_test)
                if problem_type == 'Classification':
                    results = model_evaluation_classification(results, model_name, y_train, y_pred_train, y_test, y_pred_test)
        
        # # After running all models, display results
        # for model_name, metrics in results.items():
        #     st.write(f"## {model_name}")
        #     for metric_name, metric_value in metrics.items():
        #         st.write(f"{metric_name}: {metric_value}")
                    
   
        results_df = pd.DataFrame(results).T
        st.table(results_df)
        # If you want5
     
        import plotly.express as px
   

        # Iterate over each metric in the DataFrame to create individual plots
        for metric in results_df.columns:
            # Reset index to turn the DataFrame into a long format just for the current metric
            df_metric = results_df.reset_index()[['index', metric]].rename(columns={'index': 'Model', metric: 'Value'})

            # Create the bar chart using Plotly for the current metric
            fig = px.bar(df_metric, x='Model', y='Value', color='Model', title=f'{metric} Comparison')

            # Show the plot in Streamlit
            st.plotly_chart(fig)



def main():
    st.title('🤖 Model Training')
    st.write('This is the Model Training page')

    # Load the data
    data = pd.read_csv('data/data.csv')
    columns = [column for column in data.columns]
    st.write('### Select the Target Variable')
    target_column = st.selectbox('Select the target column', data.columns)
    train_fit_models(data, columns, target_column)
  



# Assuming 'data' is your DataFrame and 'columns' is the list of column names in 'data'
    if data is not None:
        # User selects type of problem
        problem_type = st.radio('Select the problem type', ('Classification', 'Regression'))
        
        

       
       



if __name__ == '__main__':
    main()