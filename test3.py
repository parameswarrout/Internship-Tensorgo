import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from io import StringIO
import sys
import re
from Eda_plot import *

# Streamlit configuration
st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

# Styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)


class StreamlitOutputCapture:
    def __init__(self):
        self.string_io = StringIO()
        self.output_container = st.empty()

    def write(self, data):
        self.string_io.write(data)
        try:
            self.output_container.text(self.string_io.getvalue())
        except:
            pass

    def flush(self):
        pass

def execute_python_code(response):
    code_blocks = re.finditer(r'Action Input: (.*?)(?:\n|$)', response, re.DOTALL)

    for match in code_blocks:
        code = match.group(1).strip()
        st.code(code, language='python')
        st.write("Executing code:")

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            if 'df.plot' in code:
                fig, ax = plt.subplots()
                eval(code)
                st.pyplot(fig)
            elif 'plt.' in code:
                exec(code)
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
            plt.close()
            output = sys.stdout.getvalue()
            if output:
                st.write(output)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            sys.stdout = old_stdout
        break

# Sidebar for configuration
st.sidebar.title("Configuration")

# Input for Groq API Key
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
# api_key = "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"

# Model selection
model_options = ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"]
selected_model = st.sidebar.selectbox("Select a model:", model_options)

# Main content
st.title("CSV Data Analyzer")

# File uploader for CSV
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_csv(uploaded_file)

    # Input for user question
    user_question = st.text_area("Enter your question about the data:")

    # Always show the Analyze button
    analyze_button = st.button("Analyze")

    if api_key and user_question and analyze_button:
        with st.spinner("Processing..."):
            try:
                os.environ["GROQ_API_KEY"] = api_key

                # Save the uploaded file temporarily
                df.to_csv("temp.csv", index=False)

                # Create the language model
                llm = ChatGroq(model=selected_model, temperature=0)

                # Create output capture
                output_capture = StreamlitOutputCapture()

                # Create the CSV agent with verbose output
                agent_executer = create_csv_agent(
                    llm, 
                    "temp.csv", 
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )

                # Execute the query with captured output
                original_stdout = sys.stdout
                sys.stdout = output_capture
                response = agent_executer.invoke(user_question)
                sys.stdout = original_stdout

                # Display the generated response
                st.success("Generated Response:")
                st.write(response)
                execute_python_code(output_capture.string_io.getvalue())

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            finally:
                # Clean up temporary file
                if os.path.exists("temp.csv"):
                    os.remove("temp.csv")

    # Data preview and basic EDA
    st.write("Data Preview and Basic EDA:")
    basic_eda(df)

else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown("---")

