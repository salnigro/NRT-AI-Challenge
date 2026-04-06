import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader

def build_vector_store(data_path="data/cleaned_train_reviews.csv", store_path="data/faiss_index"):
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Ensure data_loader.py has been run.")
        return None

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # We load the dataframe into Document objects
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    
    print("Initializing local embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"Saving vector store to {store_path}...")
    vectorstore.save_local(store_path)
    print("Vector store built and saved.")

def load_vector_store(store_path="data/faiss_index"):
    print(f"Loading vector store from {store_path}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
