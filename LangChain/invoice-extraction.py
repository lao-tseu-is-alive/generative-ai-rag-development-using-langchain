# This RAG Application takes in an Invoice PDF, loads the data into a Vector Database,
# Send the Invoice information as context to the OpenAI GPT LLM and 
# comes back with extracted Details

import streamlit as st
from dotenv import load_dotenv
import invoiceutil as iu 

def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot...💁 ")
    st.subheader("I can help you in extracting invoice data")

    # Upload the Invoices (pdf files)
    pdf = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Extract Data")

    if submit:
        with st.spinner('Wait for it...'):
            df=iu.create_docs(pdf) 
            st.write(df)

        st.success("Hope I was able to save your time❤️")


#Invoking main function
if __name__ == '__main__':
    main()

