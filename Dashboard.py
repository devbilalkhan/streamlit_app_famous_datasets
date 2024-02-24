import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
from db.client import get_database
from db.crud import insert_documents, load_data_from_mongodb
from config import DATABASE_NAME, MODEL_RESULTS_COLLECTION
from utils import load_data, display_dataset
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
        st.write(f" #### No results found for the dataset named '{dataset_name}'.")
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

    st.write("#### Model Evaluation Results for Dataset:", dataset_name)
    st.dataframe(df.head(30), use_container_width=True)

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
        width=1200
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
    model_selected = st.selectbox("Select a model", all_models, index=0, key='model_selector')

    metrics = df.columns.drop(['Model'])
    metric_selected = metric

    df_filtered = df[df['Model'] == model_selected]

    fig = px.violin(df_filtered, y=metric_selected, title=f"Distribution of {metric_selected} for {model_selected}")
    fig2 = px.line(df_filtered, x='Run Index', y=metric_selected, title=f'{metric_selected} per Run')

    st.plotly_chart(fig)
    st.plotly_chart(fig2)


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

    # Sidebar selection for dataset
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Diamonds', 'Tips', 'Titanic'))

    if dataset_name:
        # Fetch and display model results for the selected dataset
        data = fetch_and_display_model_results(dataset_name, collection)

        # If the problem type is Regression
        if data is not None and 'ProblemType' in data.columns and data['ProblemType'].iloc[0] == 'Regression':
            st.write(f"## Graphs Metrics {dataset_name}")
            metric = st.selectbox('Select Metric', ['Test R2', 'Test MSE', 'Test RMSE', 'Test MAE'])

            # Plot model performance and metric distribution
            plot_model_performance(data, metric)
            plot_all_models_single_metric_3d(data, metric)
            plot_model_metric_distribution(data, metric)

        # If the problem type is Classification
        elif data is not None and 'ProblemType' in data.columns and data['ProblemType'].iloc[0] == 'Classification':
            st.write(f"## Graphs Metrics {dataset_name}")
            metric = st.selectbox('Select Metric', ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])

            # Plot model performance and metric distribution
            plot_model_performance(data, metric)
            plot_all_models_single_metric_3d(data, metric)
    

 
if __name__ == '__main__':
    main()