#!/usr/bin/python3
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the existing vector store with embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Define the user's query
query = "Who is Eliot?"

# Retrieve relevant documents based on query
retriever = db.as_retriever(  # Fixed: db.as.retriever -> db.as_retriever
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.1
    }
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")  # Fixed: \--- -> ---
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")  # Added error handling for metadata
    print("-" * 50)  # Separator between documents