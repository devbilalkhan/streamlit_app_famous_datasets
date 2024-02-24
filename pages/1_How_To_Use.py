
import streamlit as st
def main():

  st.title('How to Use the Guide')

  st.markdown("""
  Welcome to the Machine Learning Dashboard! This interactive application guides you through the process of data cleaning, model training, and performance evaluation with datasets like Diamonds, Iris, Tips, and Titanic. Follow the steps below to get started.
  """)

  st.markdown("## Step-by-Step Guide")

  st.markdown("""
  #### 🧹 Step 1: Data Cleaning
  1. **Select a Dataset**: From the 'Data Cleaning' section, choose a dataset from the dropdown menu.
  2. **Inspect Data**: Examine the dataset for missing values or inconsistencies.
  3. **Handle Missing Values**: Opt to remove or impute missing data using various strategies.
  4. **Drop Columns**: Eliminate irrelevant features by dropping columns.
  5. **Scale Data**: Standardize or normalize your data if necessary for your model.

  #### 🤖 Step 2: Model Training
  1. **Choose a Model**: In the 'Model Training' section, pick a machine learning model from the list.
  2. **Configure Split**: Decide the train-test split ratio for your data.
  3. **Train Model**: Hit the 'Train' button to start training your model on the preprocessed data.
  4. **Monitor Progress**: Keep an eye on the training progress through the interface.
  5. **Run Multiple Models**: Compare different models by training them one after the other.

  #### 📊 Step 3: Dashboard Metrics
  1. **Navigate to Dashboard**: After training, switch over to the 'Dashboard' to view metrics.
  2. **Select Dataset**: Compare different models by selecting them in the dashboard for any dataset.
  3. **Review Metrics**: Analyze accuracy, precision, recall, F1 score, and more.
  4. **Utilize Visualization**: Employ the integrated visualization tools for deeper insight.

  #### 🎨 Step 4: Data Visualization
  1. **Open Visualization Section**: Explore further with the 'Data Visualization' section.
  2. **Create Plots**: Generate histograms, box plots, scatter plots, and more to understand your data.
  3. **Interactive Exploration**: Dive deeper into the plots with interactive features.

  #### 💡 Tips for Effective Use
  - **Save Your Work**: While the app doesn't save your session, you can download your results.
              Working to integerate with a cloud storage service for saving your work.
  - **Experiment Freely**: Test various preprocessing and model training options.
  - **Provide Feedback**: Use the in-app form to send feedback or report issues (Coming Soon).
  """)

  st.markdown("## Need Help?")
  st.markdown("""
  If you require assistance or further information, please ask me on discord `@devbilalkhan`. Happy analyzing!
  """)

if __name__ == "__main__":
    main()
