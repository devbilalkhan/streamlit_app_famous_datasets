import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd

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

def main():

    # Start of your Streamlit app
    app_name = "DataGem Analytics Suite"
    # Dictionary of pages
    st.set_page_config(
        page_title="Dashboard",
        page_icon=":shark:",
      
    )
    
    st.title(app_name)
    # Description or introduction to your app
    st.markdown("""
        Welcome to **DataGem Analytics Suite**, an integrated platform for data exploration, 
        cleaning, and model training across a variety of classic datasets.
    """)

    # Load the metrics from the CSV file
    results_df = pd.read_csv('data/model_metrics.csv')
    problem_type = results_df.columns[0]
    results_df = results_df.rename(columns={problem_type: 'Model'})
    # Round the values to 3 decimal places
    results_df = results_df.round(3)

    if problem_type == 'Classification':

        plot_all_metrics_line(results_df)

        plot_radar_chart(results_df)
        plot_heatmap(results_df)
        metrics_bars_plot(results_df)
    else: 
        # # Call the line plot function for 'Test MSE'
        line_plot(results_df, 'Test MSE')
         # Call the 3D scatter plot function for three metrics
        scatter_3d_plot(results_df, 'Test MSE', 'Test RMSE', 'Test MAE')
        # Call the bar plot function for 'Test RMSE'
        bar_plot(results_df, 'Test RMSE')

        # Call the scatter plot matrix function for the entire DataFrame
        splom_plot(results_df)
        metrics_bar_plots(results_df)
     

if __name__ == '__main__':
    main()