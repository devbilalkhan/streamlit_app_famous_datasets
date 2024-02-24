import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
from db.client import get_database
from db.crud import insert_documents, load_data_from_mongodb
from config import DATABASE_NAME, MODEL_RESULTS_COLLECTION
from utils import load_data, display_dataset
from pymongo import MongoClient



def fetch_and_display_model_results(dataset_name, collection):
    # Query the database for records with the dataset name
    query = {'DatasetName': dataset_name}
    records = collection.find(query)

    # Convert the records to a DataFrame
    df = pd.DataFrame(list(records))
     # Check if any records were found
    if df.empty:
        st.write(f" #### No results found for the dataset named '{dataset_name}'.")
        return
    # Optionally, if there's a timestamp field, you can sort the results by that field
    timestamp_field = 'createdAt'  # Replace with your actual timestamp field name
    if timestamp_field in df.columns:
        df[timestamp_field] = pd.to_datetime(df[timestamp_field])
        df.sort_values(by=timestamp_field, ascending=False, inplace=True)

    # Drop the '_id' column if you don't want to display it
    df.drop(columns=['_id', 'DatasetName'], inplace=True)
     # Check if the DataFrame has a 'Classification' or 'Regression' column and filter
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

            
    # Display the DataFrame in Streamlit
    st.write("#### Model Evaluation Results for Dataset:", dataset_name)
    st.dataframe(df.head(30), use_container_width=True)

    # show total number of record 
    st.write(f"Total number of records: {len(df)}")
    
    return df



TASK_STATE_KEY = 'task_type'

custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                     '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

def metrics_bars_plot(results_df):
    # Iterate over the metrics
    for i in range(1, len(results_df.columns)):  # Start from 1 to skip the 'Model' column
        metric = results_df.columns[i]
        df_metric = results_df[['Model', metric]].rename(columns={metric: 'Value'})
        fig = px.bar(df_metric, x='Model', y='Value', color='Model', title=f'{metric} Comparison')

        # Display the plot in its own row
        st.plotly_chart(fig, use_container_width=True)


def plot_all_metrics_line(data):
    melted_data = data.melt(id_vars=['Model'], var_name='Metric', value_name='Value')
    fig = px.line(melted_data, x='Model', y='Value', color='Metric', markers=True, title='Model Performance Metrics')
    st.plotly_chart(fig)

def plot_radar_chart(data):
    fig = px.line_polar(data, r='Test Accuracy', theta='Model', line_close=True,
                        title='Model Performance Comparison (Accuracy)')
    for metric in ['Test Precision', 'Test Recall', 'Test F1']:
        fig.add_trace(px.line_polar(data, r=metric, theta='Model', line_close=True).data[0])
    fig.update_traces(fill='toself', fillcolor=custom_colors[3])
    st.plotly_chart(fig)



def plot_heatmap(data):
    metric_data = data.drop('Model', axis=1)

    
    fig = px.imshow(metric_data, x=metric_data.columns, y=data['Model'], aspect='auto',
                    labels=dict(x='Metric', y='Model', color='Performance'),
                    title='Heatmap of Model Performance Metrics',
              
                    )
    st.plotly_chart(fig)

# Function to plot bar plots for each metric
def metrics_bar_plots(df):
    metrics = df.columns[1:]  # Exclude the 'Model' column
    for metric in metrics:
        fig = px.bar(df, x='Model', y=metric, text=metric, color='Model', title=f'{metric} Comparison')
        st.plotly_chart(fig, use_container_width=True)

def line_plot(df, column):
    fig = px.line(df, y=column, title=f'{column} per Model', markers=True)
    fig.update_xaxes(title_text='Model')
    fig.update_yaxes(title_text=column)
    st.plotly_chart(fig)
    

def bar_plot(df, column):
    fig = px.bar(df, y=column, title=f'{column} Comparison', text=column)
    fig.update_xaxes(title_text='Model')
    fig.update_yaxes(title_text=column)
    st.plotly_chart(fig)


def scatter_3d_plot(df, x, y, z):
    fig = px.scatter_3d(df, x=x, y=y, z=z, title=f'3D Scatter of {x}, {y}, and {z}')
    fig.update_layout(scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z))
    st.plotly_chart(fig)

def splom_plot(df):
    fig = px.scatter_matrix(df,
                            dimensions=df.columns,
                            title='Scatter Matrix of Metrics')
    fig.update_traces(diagonal_visible=False)

    # Rotate labels if necessary
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=45)
    )
    
    # Decrease tick font size
    fig.update_layout(
        xaxis_tickfont_size=10,
        yaxis_tickfont_size=10
    )

    # Optionally, update the layout to increase figure size
    fig.update_layout(
        width=1200,  # Adjust the width of the figure
        height=800,  # Adjust the height of the figure
    )

    # Use this to display the plot in a Streamlit app
    st.plotly_chart(fig)



def scatter_3d_plot(df, x, y, z):
    fig = px.scatter_3d(df, x=x, y=y, z=z, title=f'3D Scatter of {x}, {y}, and {z}')

    # Convert hex color to RGBA for Plotly
    bg_color = 'rgb(38,39,48)'
    line_color = 'white'

    fig.update_layout(scene=dict(
                    xaxis=dict(
                        title=x,
                        backgroundcolor=bg_color,
                        gridcolor=line_color,
                        showbackground=True,
                        zerolinecolor=line_color,),
                    yaxis=dict(
                        title=y,
                        backgroundcolor=bg_color,
                        gridcolor=line_color,
                        showbackground=True,
                        zerolinecolor=line_color,),
                    zaxis=dict(
                        title=z,
                        backgroundcolor=bg_color,
                        gridcolor=line_color,
                        showbackground=True,
                        zerolinecolor=line_color,),
                    ),
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color)
    st.plotly_chart(fig)


def plot_all_models_single_metric_3d(df, metric):
    st.write("## 3D Scatter Plot")

    # Create a new column 'Run Index' which serves as a simple counter for each run
    df['Run Index'] = range(len(df))
    # Convert hex color to RGBA for Plotly
    bg_color = 'rgb(38,39,48)'
    line_color = 'white'
    # Generate the 3D scatter plot for the selected metric
    fig = px.scatter_3d(df, x='Run Index', y=metric, z='Model', color='Model',
                        title=f"3D Scatter Plot of {metric} Across All Models")

    # Update layout for better visual on dark screens
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

    # Display the plot
    st.plotly_chart(fig)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_model_performance(model_records, metric):
   
    df = pd.DataFrame(model_records)
    
    # Group by 'Model' and calculate the average for the specified metric
    df_metrics = df.groupby('Model').agg({metric: 'mean'}).reset_index()
    
    # Create subplots: one for bar chart and one for line chart for the specified metric
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Average {metric} Bar Chart', f'Average {metric} Line Chart'))
    
    # Add bar chart for the specified metric in the first subplot
    fig.add_trace(
        go.Bar(name=metric, x=df_metrics['Model'], y=df_metrics[metric]),
        row=1, col=1
    )
    
    # Add line chart for the specified metric in the second subplot
    fig.add_trace(
        go.Scatter(name=metric, x=df_metrics['Model'], y=df_metrics[metric], mode='lines+markers'),
        row=1, col=2
    )
    
    # Update layout for readability
    fig.update_layout(
        title_text=f"Model Performance - {metric} Comparison",
        barmode='group',
        height=600,
        width=1200
    )

    st.plotly_chart(fig)

def plot_model_metric_distribution(df, metric):
   # Check if '_id' column exists and drop it if it does
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    
    st.write(f"## Metic Distribution ")
    # Model selection
    all_models = df['Model'].unique().tolist()
    default_model = all_models[0]  # The first model in the list will be the default
    model_selected = st.selectbox("Select a model", all_models, index=0, key='model_selector')

    # Metric selection
    # Assuming 'Model' is a column in df
    metrics = df.columns.drop(['Model'])  # Dropping 'Model' column to get metric names
    metric_selected = metric

    # Filter the DataFrame based on the selected model
    df_filtered = df[df['Model'] == model_selected]

    # Generate the box plot for the selected metric
    fig = px.violin(df_filtered, y=metric_selected, title=f"Distribution of {metric_selected} for {model_selected}")

 
    fig2 = px.line(df_filtered, x='Run Index', y=metric_selected, title=f'{metric_selected} per Run')
    # Display the plot
    st.plotly_chart(fig)
    st.plotly_chart(fig2)



def main():
    
    app_name = "DataGem Analytics Suite"
    # Dictionary of pages
    st.set_page_config(
        page_title="Dashboard",
        page_icon=":shark:",
        # layout="wide",
    )
    # Use markdown to create a horizontal line as a divider
    st.title(app_name)
    # Description or introduction to your app
    st.markdown("""
        Welcome to **DataGem Analytics Suite**, an integrated platform for data exploration, 
        cleaning, and model training across a variety of classic datasets.
    """)
    db = get_database(DATABASE_NAME)
    collection = db[MODEL_RESULTS_COLLECTION]

    
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Diamonds', 'Tips', 'Titanic'))
    # record = collection.find({'DatasetName': dataset_name})
    # st.dataframe(record)
    
    
    # print the first line of the data 
 
    #     # Check if 'ProblemType' is one of the columns in the DataFrame
    # if column_name in data.columns:
    #     # Check if the problem type for the current dataset is 'regression'
    #     if data[column_name].iloc[0] == 'Classification':
    #         # When a dataset with 'regression' problem type is selected,
    #         # fetch and display the model results for that dataset
    #         if dataset_name:            
    #             data = fetch_and_display_model_results(dataset_name)
                
 
    if dataset_name:
        # data = load_data_from_mongodb(MODEL_RESULTS_COLLECTION, 'Regression')
        data = fetch_and_display_model_results(dataset_name, collection)
     
        if data is not None and \
            'ProblemType' in data.columns and \
            data['ProblemType'].iloc[0] == 'Regression':
      
            st.write(f"## Graphs Metrics {dataset_name}")
            
            # Give user option to select each metric and save it in a variable
            metric = st.selectbox('Select Metric', ['Test R2', 'Test MSE', 'Test RMSE', 'Test MAE'])
            
            # The following functions should be defined elsewhere in your code
            # They will plot the model performance based on the selected metric
            plot_model_performance(data, metric)

            plot_all_models_single_metric_3d(data, metric)
            plot_model_metric_distribution(data, metric)

        elif data is not None and \
        'ProblemType' in data.columns and \
            data['ProblemType'].iloc[0] == 'Classification':
            st.write(f"## Graphs Metrics {dataset_name}")
            # Give user option to select each metric and save it in a variable
            metric = st.selectbox('Select Metric', ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
            
            # The following functions should be defined elsewhere in your code
            # They will plot the model performance based on the selected metric
            plot_model_performance(data, metric)

            plot_all_models_single_metric_3d(data, metric)
            
    

 
if __name__ == '__main__':
    main()