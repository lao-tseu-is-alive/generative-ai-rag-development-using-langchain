from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import csv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Import Google API Key
from dotenv import load_dotenv
import os
load_dotenv()

# Set yout LLM to Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-pro",  google_api_key=os.getenv("GOOGLE_API_KEY"))


# Define the prompt template
template = """
You are creating test data for an employee database.
Generate information for {num_employees} employees with the following fields, separated by commas:

Employee Id, Name, Department, Basic Salary, Incentives, Date of Joining (YYYY-MM-DD)

Make sure the data is diverse and realistic.
"""
prompt = PromptTemplate(template=template, input_variables=["num_employees"])

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("Employee Test Data Generator")

num_employees = st.number_input("Number of employees", min_value=1, max_value=100, value=20)

if st.button("Generate Data"):
    # Run the chain to generate the data
    data = chain.run({"num_employees": num_employees})

    # Process the output and write to CSV
    with open('employee_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Employee Id", "Name", "Department", "Basic Salary", "Incentives", "Date of Joining"])
        for line in data.strip().split('\n'):
            writer.writerow(line.split(','))

    st.download_button(
        label="Download CSV",
        data=open('employee_data.csv', 'r').read(),
        file_name='employee_data.csv',
        mime='text/csv'
    )
