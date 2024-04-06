import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template


# Function to read from pdfs and extract the text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Reading each PDF
        pdf_reader = PdfReader(pdf)
        # Extracting text from each page
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def stop_at_word(input_string, stop_word):
    words = input_string.split()
    result = []
    for word in words:
        if word == stop_word:
            break
        result.append(word)
    return ' '.join(result)


# Dividing the text in different chunks (groups)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,  # 1000 characters
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# converting the text chunks into embeddings and storing inside vectore database
def get_vectorestore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})
    vectorestore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)
    return vectorestore


def get_conversation_chain(vector_store):
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 128, 'temperature': 0.01})
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                               retriever=vector_store.as_retriever(
                                                                   search_kwargs={"k": 2}),
                                                               memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        message = stop_at_word(message.content, 'Unhelpful')
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message),
                     unsafe_allow_html=True)


def main():
    load_dotenv()
    # Title for the application

    st.write(css, unsafe_allow_html=True)

    # to make it persistent
    # since streamlit reinitializes the
    # initializing session_state

    st.header("Chat with Documents 📄")

    if "conversation" not in st.session_state:
        st.write(user_template.replace(
            "{{MSG}}", "Hello Bot👋"), unsafe_allow_html=True)
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.write(bot_template.replace(
            "{{MSG}}", "Hello Human 😁"), unsafe_allow_html=True)
        st.session_state.chat_history = None

    user_question = st.text_input("Ask Question about your document: ")

    if user_question:
        handle_userinput(user_question)

    # The sidebar with text, button and upload option
    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDF", accept_multiple_files=True)
        if st.button("Process"):
            # spinner to make it user friendly
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # turn it into a vectorstore
                vectorestore = get_vectorestore(text_chunks)

                # create converstion chain
                st.session_state.conversation = get_conversation_chain(
                    vectorestore)


if __name__ == '__main__':
    main()
