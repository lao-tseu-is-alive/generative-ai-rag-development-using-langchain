# This program is intended to create a Chatbot that accesses a FAISS Vector database that contains a large HR website 
# with tons of HR policies, practices and domain knowledge. The ChatBot will give the user the ability to query on any 
# HR related information in a conversation form with conversational meory like Chat GPT. 
# The UI of the Chat Bot is done using the Streamlit Library.

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def build_chat_history(chat_history_list):
    # This function takes in the Chat History Messages in a List of Tuples format
    # and turns it into a series of Human and AI Message objects
    chat_history = []
    for message in chat_history_list:        
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))
    
    return chat_history

def query(question, chat_history):
    """
    This function does the following:
    1. Receives two parameters - 'question' - a string and 'chat_history' - a Python List of tuples containing accumulating question-answer pairs    
    2. Load the local FAISS database where the entire website is stored as Embedding vectors
    3. Create a ConversationalBufferMemory object wth 'chat_history'
    4. Create a ConversationalRetrievalChain object with the FAISS DB as the Retriever (LLM lets us create Reriever objects against data stores)
    5. Invoke the Retriever object with the Query and Chat History
    6. Returns the response
    """

    chat_history = build_chat_history(chat_history)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("HR-ChatBot/faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, new_db.as_retriever(), condense_question_prompt
    )
    system_prompt = (
        "You are an assistant for question-answering tasks on HR Policy. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )

def show_ui():
    """
    This function does the following:
    1. Implements the Streamlim UI
    2. Implements two session_state vatiables - 'messages' - to contain the accumulating Questions and Answers to be displayed on the UI and
       'chat_history' - the accumulating question-answer pairs as a List of Tuples to be served to the Retriever object as chat_history
    3. For each user query, the response is obtained by invoking the 'query' function and the chat histories are byilt up   
    """
    st.title("Yours Truly Human Resources Chatbot")    
    st.image("HR-ChatBot/c4x-cbt.png")
    st.subheader("Please enter your HR Query ")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your HR Policy related Query: "):
        # Invoke the function with the Retriver with chat history and display responses in chat container in question-answer pairs 
        with st.spinner("Working on your query...."):     
            response = query(question=prompt, chat_history=st.session_state.chat_history)            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])    

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

# Program Entry.....
if __name__ == "__main__":
    show_ui() 
    