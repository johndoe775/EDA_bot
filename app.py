import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()  # This will load environment variables from the .env file

api_key = os.getenv("api_key")

# Initialize LangChain and LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Specify the correct model
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=api_key,  # Ensure the API key is set
)

# Function to get the recommended charts using LangChain's Google API
def get_recommended_graphs(df, purpose):
    # Extract DataFrame info
    df_info = df.info()

    # Extract DataFrame column data types
    df_dtypes = df.dtypes.to_string()  # Convert dtypes to string format for easy display

    # Define the prompt template
    prompt_template = """
    Given the following dataframe info, column data types, and analysis purpose, suggest the most suitable charts from Seaborn for EDA.

    ### DataFrame Info:
    {df_info}

    ### DataFrame Data Types:
    {df_dtypes}

    ### Purpose of the Analysis:
    {purpose}

    Do not suggest charts for columns that are not in the dataframe, and ensure that Column 1 and Column 2 are from the given dataframe columns. Don't hallucinate the column names.

    Based on this, recommend the appropriate chart types and corresponding columns in the following format and show only these and nothing else:
    Column 1, Column 2, Chart Type

    Analysis type must be univariate or bivariate and nothing else.
    Further if they are any box plots replace them with bar plots.
    Scatter plots or univariate analysis can be rejected.

    Only suggest valid Seaborn charts based on the data available.
    """

    # Initialize the prompt template
    prompt = PromptTemplate(
        input_variables=["df_info", "df_dtypes", "purpose"],
        template=prompt_template
    )

    # Create an LLMChain using the prompt and the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # Construct the prompt input
    prompt_input = prompt.format(df_info=df_info, df_dtypes=df_dtypes, purpose=purpose)

    # Generate the recommendation using LangChain
    response = chain.run(df_info=df_info, df_dtypes=df_dtypes, purpose=purpose)

    return response

# Streamlit UI setup
st.title("Data Analysis with LangChain & Seaborn")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Data preview: {df.head()}")

    # Purpose input field
    purpose = st.text_input("Enter the analysis purpose:", "Assess the metrics of plans")

    # Get recommended graphs based on the uploaded data and purpose
    if st.button("Get Recommended Graphs"):
        recommended_graphs = get_recommended_graphs(df, purpose)

        st.subheader("Recommended Graphs based on Purpose:")
        st.write(recommended_graphs)

        # Extract graph suggestions
        graph_suggestions = [line.strip() for line in recommended_graphs.split('\n') if line.strip()]
        st.write("Graph Suggestions:", graph_suggestions)

        # Function to dynamically plot graphs based on AI suggestions
        def plot_graphs_in_grid(graph_suggestions):
            num_graphs = len(graph_suggestions)
            rows = (num_graphs // 3) + (1 if num_graphs % 3 else 0)  # Create enough rows for 3 columns

            # Create a grid of subplots
            fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))

            # Flatten the axes for easy indexing
            axes = axes.flatten()

            for i, suggestion in enumerate(graph_suggestions):
                # Parse the suggestion: Column1, Column2, ChartType
                columns_and_chart = suggestion.split(',')
                if len(columns_and_chart) == 3:
                    column1 = columns_and_chart[0].strip()
                    column2 = columns_and_chart[1].strip()
                    chart_type = columns_and_chart[2].strip()

                    # Dynamically call the plotting function using getattr
                    if chart_type in dir(sns):  # Check if chart_type exists as an attribute in seaborn
                        plot_function = getattr(sns, chart_type)

                        # Handle the case where column1 exists in df
                        if column1 in df.columns:
                            # Plot in the current axis
                            ax = axes[i]

                            if chart_type == "countplot":
                                plot_function(data=df, x=column1, ax=ax)
                            else:
                                if column2 in df.columns:
                                    plot_function(data=df, x=column1, y=column2, ax=ax)
                                else:
                                    st.write(f"Invalid columns for {chart_type}: {column1}, {column2}")

                            # Set the title for the plot
                            ax.set_title(f'{chart_type.capitalize()}: {column1} vs {column2}')
                            ax.legend(title='Legend', loc='upper right')

            # Adjust layout and show the plots
            plt.tight_layout()
            st.pyplot(fig)

        # Plot the recommended graphs in a grid layout
        plot_graphs_in_grid(graph_suggestions)
