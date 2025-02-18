from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import os 
import streamlit as st
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:\\redener\\MynexRenpro\\Langchain\\.env")

## load the Groq API key
# api_key = os.getenv("Nivida_key")

#as we need to read mulitple pdf documnets 
def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=NVIDIAEmbeddings(api_key="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")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
        print("hEllo")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


prompt1=st.text_input("Enter Your Question From Documents")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    print("Document Chain Is Ready")
    retriever=st.session_state.vectors.as_retriever()
    print("Retriever Is Ready")
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    print("Retrieval Chain Is Ready")
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("haha")
    print("Response time :",time.process_time()-start)
    print(response)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
