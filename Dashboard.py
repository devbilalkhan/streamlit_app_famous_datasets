import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
from db.client import get_database
from db.crud import get_dataset_name, delete_collection
from config import DATABASE_NAME, MODEL_RESULTS_COLLECTION, DATASET_COLLECTION_NAME
from utils import load_data, clean_dataset_name

from pymongo import MongoClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fetch_and_display_model_results(dataset_name, collection):
    """
    Fetches and displays model results from a MongoDB collection for a given dataset.

    Parameters:
    dataset_name (str): The name of the dataset to fetch results for.
    collection (pymongo.collection.Collection): The MongoDB collection to fetch results from.

    Returns:
    df (pandas.DataFrame): A DataFrame containing the fetched results.
    """
    query = {'DatasetName': dataset_name}
    records = collection.find(query)

    df = pd.DataFrame(list(records))

    if df.empty:
        st.write(f" #### No results found for the dataset named {dataset_name.title()}.")
        return

    timestamp_field = 'createdAt'
    if timestamp_field in df.columns:
        df[timestamp_field] = pd.to_datetime(df[timestamp_field])
        df.sort_values(by=timestamp_field, ascending=False, inplace=True)

    df.drop(columns=['_id', 'DatasetName'], inplace=True)

    if 'Classification' in df.columns:
        model_names = sorted(df['Classification'].unique())
        selected_models = st.multiselect('Filter by Classification models:', model_names)
        if selected_models:
            df = df[df['Classification'].isin(selected_models)]
    if 'Regression' in df.columns:
        model_names = sorted(df['Regression'].unique())
        selected_models = st.multiselect('Filter by Regression models:', model_names)
        if selected_models:
            df = df[df['Regression'].isin(selected_models)]

    st.write("##### Model Evaluation Results for Dataset:", dataset_name.title())
    df_filled = df.fillna('-')
    st.dataframe(df_filled, use_container_width=True)

    st.write(f"Total number of records: {len(df)}")
    
    return df



def plot_all_models_single_metric_3d(df, metric):
    """
    Generates a 3D scatter plot of a single metric across all models.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    metric (str): The metric to be plotted.

    Displays:
    A 3D scatter plot in the Streamlit app.
    """
    df['Run Index'] = range(len(df))
    bg_color = 'rgb(38,39,48)'
    fig = px.scatter_3d(df, x='Run Index', y=metric, z='Model', color='Model',
                        title=f"3D Scatter Plot of {metric} Across All Models")
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(backgroundcolor=bg_color),
            yaxis=dict(backgroundcolor=bg_color),
            zaxis=dict(backgroundcolor=bg_color),
            xaxis_title='Run Index',
            yaxis_title=metric,
            zaxis_title='Model'
        ),
        width=1200,
        height=600
    )
    st.plotly_chart(fig)


def plot_model_performance(model_records, metric):
    """
    Plots the performance of different models based on a specified metric.

    Parameters:
    model_records (list): A list of dictionaries where each dictionary represents a model's performance record.
    metric (str): The performance metric based on which the models' performance is to be plotted.

    Displays:
    A bar chart and a line chart comparing the models' performance based on the specified metric.
    """
    df = pd.DataFrame(model_records)    
    df_metrics = df.groupby('Model').agg({metric: 'mean'}).reset_index()
  
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Average {metric} Bar Chart', f'Average {metric} Line Chart'))
    fig.add_trace(
        go.Bar(name=metric, x=df_metrics['Model'], y=df_metrics[metric]),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(name=metric, x=df_metrics['Model'], y=df_metrics[metric], mode='lines+markers'),
        row=1, col=2
    )
    fig.update_layout(
        title_text=f"Model Performance - {metric} Comparison",
        barmode='group',
        height=600,
        width=800
    )
    st.plotly_chart(fig)


def plot_model_metric_distribution(df, metric):
    """
    Plots the distribution of a specified metric for a selected model.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    metric (str): The metric whose distribution is to be plotted.

    Displays:
    A violin plot and a line plot showing the distribution of the specified metric for the selected model.
    """
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    st.write(f"## Metric Distribution ")
    all_models = df['Model'].unique().tolist()
    default_model = all_models[0]
    model_selected = st.selectbox("Select a model", all_models, index=0, key='model_selector_1')

    metrics = df.columns.drop(['Model'])
    metric_selected = metric

    df_filtered = df[df['Model'] == model_selected]

    fig = px.violin(df_filtered, y=metric_selected, title=f"Distribution of {metric_selected} for {model_selected}")

    st.plotly_chart(fig)

   
def plot_model_metric_distribution2(df, metric):
    """
    Plots the distribution of a specified metric for a selected model.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    metric (str): The metric whose distribution is to be plotted.

    Displays:
    A violin plot and a line plot showing the distribution of the specified metric for the selected model.
    """
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    st.write(f"## Metric Distribution ")
    all_models = df['Model'].unique().tolist()
    default_model = all_models[0]
    model_selected = st.selectbox("Select a model", all_models, index=0, key='model_selector_2')

    metrics = df.columns.drop(['Model'])
    metric_selected = metric

    df_filtered = df[df['Model'] == model_selected]
  
    fig = px.line(df_filtered, x='Run Index', y=metric_selected, title=f'{metric_selected} per Run')
    st.plotly_chart(fig)

def select_existing_datasets(collection):
    """
    Displays a selectbox with the names of existing datasets and returns the selected dataset.

    Parameters:
    collection (str): The name of the collection to get the dataset names from.

    Returns:
    str: The name of the selected dataset.
    """
    dataset_names = get_dataset_name(collection)
    default_datasets = ['Iris', 'Diamonds', 'Tips', 'Titanic']
    if len(dataset_names) == 0:
        
        # Sidebar selection for datasets
        dataset_selected = st.sidebar.selectbox('Select Dataset or upload one', default_datasets)
    else:
        dataset_names = [name.title() for name in dataset_names]
        dataset_names.extend(default_datasets)
        dataset_selected = st.sidebar.selectbox('Select a dataset', dataset_names)
    return dataset_selected


def plot_learning_rate_vs_metric(df, lr_column='learning_rate', metric_column='Test R2'):
    """
    Plots a graph of learning rate against a specified metric for a given model.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the model data.
    model_name (str): The name of the model to filter by.
    lr_column (str): The name of the column containing learning rate data. Defaults to 'learning_rate'.
    metric_column (str): The name of the column containing the metric to plot against learning rate. Defaults to 'test R2'.
    """
    # Check if the learning rate and metric columns exist
    if lr_column not in df.columns or metric_column not in df.columns:
        raise ValueError(f"Columns {lr_column} and/or {metric_column} not found in DataFrame.")
    
    # Drop rows where either learning rate or metric column has NaN values
    filtered_df = df.dropna(subset=[lr_column, metric_column])
    
    # Create the scatter plot for all models
    fig = px.line(filtered_df, x=lr_column, y=metric_column, color='Model', 
                     title=f'Learning Rate vs {metric_column} for All Models')
    
    st.plotly_chart(fig)

# main
def main():
    app_name = "DataGem Analytics Suite"
    st.set_page_config(page_title="Dashboard", page_icon=":shark:")

    st.title(app_name)

    # Introduction to the app
    st.markdown("""
        Welcome to **DataGem Analytics Suite**, an integrated platform for data exploration, 
        cleaning, and model training across a variety of classic datasets.
    """)

    # Connect to the database
    db = get_database(DATABASE_NAME)
    collection = db[MODEL_RESULTS_COLLECTION]

    dataset_name = select_existing_datasets(DATASET_COLLECTION_NAME)
    dataset_name = clean_dataset_name(dataset_name.lower())   
    
    data = pd.read_csv(f'data/{dataset_name}.csv')
   
    # if st.sidebar.button('Delete Dataset COLLECTION'):
    #     delete_collection(DATASET_COLLECTION_NAME)

    if dataset_name:
        # Fetch and display model results for the selected dataset
        data = fetch_and_display_model_results(dataset_name, collection)

        # If the problem type is Regression
        if data is not None and 'ProblemType' in data.columns and data['ProblemType'].iloc[0] == 'Regression':
            st.write(f"## Graphs Metrics {dataset_name}")
            metric = st.selectbox('Select Metric', ['Test R2', 'Test MSE', 'Test RMSE', 'Test MAE'], key='clf_metric_selector')
            print(data.head())
            # Plot model performance and metric distribution
            #plot_model_performance(data, metric)
            plot_all_models_single_metric_3d(data, metric)
            plot_model_metric_distribution(data, metric)
            plot_model_metric_distribution2(data, metric)
            plot_learning_rate_vs_metric(data)
            

        # If the problem type is Classification
        elif data is not None and 'ProblemType' in data.columns and data['ProblemType'].iloc[0] == 'Classification':
            st.write(f"## Graphs Metrics {dataset_name}")
            metric = st.selectbox('Select Metric', ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'], key='rg_metric_selector')

            # Plot model performance and metric distribution
            #plot_model_performance(data, metric)
            plot_all_models_single_metric_3d(data, metric)   

 
if __name__ == '__main__':
    main()