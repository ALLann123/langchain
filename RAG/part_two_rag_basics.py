#!/usr/bin/python3
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the persistent directory (must match the one used during vector creation)
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load embeddings with proper settings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}  # Chroma handles normalization
)

# Load Chroma
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Define your semantic query
query = "Who had gone off to the Ethiopians?"

# Use a retriever to search for relevant chunks
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# Retrieve and print matching documents
relevant_docs = retriever.invoke(query)

print("----[+] Relevant Documents ----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
