# NVIDIA AI-Powered Document Chatbot
This project is a Streamlit-based chatbot application that enables users to interact with documents using NVIDIA AI models. It supports vector embedding of multiple documents and allows for intelligent querying with an LLM from NVIDIA.

## Features
Chat with Documents: Query information from uploaded PDFs.
NVIDIA AI Integration: Utilizes NVIDIA's Llama-3.1-Nemotron-70b-instruct model.
Streamlit UI: User-friendly interface for document interaction.
Vector Embeddings: Converts documents into embeddings using NVIDIA AI.
FAISS Vector Store: Stores document embeddings for efficient retrieval.

## Requirements
Ensure you have the following installed:
Python 3.x
Streamlit
LangChain
FAISS
PyPDFDirectoryLoader

## Configuration
PDF Loading: Loads multiple PDF documents from a specified directory.
Vector Embedding: Converts documents into vector embeddings for retrieval.

## Usage
Upload PDF documents for processing.
Click "Documents Embedding" to store vector representations.
Enter a query related to the uploaded documents.
Receive intelligent responses based on document content.
