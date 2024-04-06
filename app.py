import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template

# load the pdf
pdf_path = 'data/'
loader = DirectoryLoader(pdf_path, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

# create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(
                                                  search_kwargs={"k": 2}),
                                              memory=memory)


# Purpose:
# ---
# This script initializes a Streamlit application to run a Mental Health ChatBot.
# It loads documents, preprocesses them, creates embeddings, sets up a conversational retrieval chain,
# and provides a user interface for interacting with the ChatBot.

# Input Arguments:
# ---
# None

# Returns:
# ---
# None

# Example call:
# ---
# The script is meant to be run as a Streamlit application.
st.title("Mental Health ChatBot 🧑🏽‍⚕️")
st.write(css, unsafe_allow_html=True)


# Purpose:
# ---
# This function stops at a specified word in an input string and returns the string up to that word.

# Input Arguments:
# ---
# `input_string`: The input string to be processed.
# `stop_word`: The word at which to stop processing the input string.

# Returns:
# ---
# `result`: The processed string up to the stop word.

# Example call:
# ---
# processed_text = stop_at_word(input_string, 'stop_word')
def stop_at_word(input_string, stop_word):
    words = input_string.split()
    result = []
    for word in words:
        if word == stop_word:
            break
        result.append(word)
    return ' '.join(result)


# Purpose:
# ---
# This function handles the conversation between the user and the ChatBot.
# It queries the conversational retrieval chain and updates the session state with the conversation history.

# Input Arguments:
# ---
# `query`: The user's query or message.

# Returns:
# ---
# `result['answer']`: The ChatBot's response to the user's query.

# Example call:
# ---
# response = conversation_chat(user_query)
def conversation_chat(query):
    result = chain(
        {"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    st.session_state.chat_history = result['answer']
    return result['answer']


# Purpose:
# ---
# This function initializes the session state for the Streamlit application.
# It initializes session variables such as chat history and past interactions.

# Input Arguments:
# ---
# None

# Returns:
# ---
# None

# Example call:
# ---
# initialize_session_state()
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.write(user_template.replace(
            "{{MSG}}", "Hello Bot👋"), unsafe_allow_html=True)
        st.write(bot_template.replace(
            "{{MSG}}", "Hello Human 😁"), unsafe_allow_html=True)

    st.session_state['generated'] = []
    st.session_state['past'] = []


# Purpose:
# ---
# This function displays the chat history in the Streamlit application.
# It allows users to input questions and see the conversation history with the ChatBot.

# Input Arguments:
# ---
# None

# Returns:
# ---
# None

# Example call:
# ---
# display_chat_history()
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input(
                "Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message = st.session_state["generated"][i]
            message = stop_at_word(message, 'Unhelpful')
            st.write(user_template.replace(
                "{{MSG}}", st.session_state["past"][i]), unsafe_allow_html=True)
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
