#!/usr/bin/python3
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from langchain_openai

# Define the directory containing the file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the vector store already exists
if not os.path.exists(persistent_directory):
    print("[+] Persistent directory does not exist. Initializing vector store....")
    
    # Verify the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read and split the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # Recommended to have some overlap
        separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # Display information
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content[:200]}...\n")  # Show first 200 chars

    # Create embeddings
    print("===================================================")
    print("\n--- Creating Embeddings ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Specify device
        encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
    )
    
    # Create and persist the vector store
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    
    # Explicitly persist (good practice)
    db.persist()
    print("\n--- Vector store created and persisted ---")

else:
    print("Vector store already exists. No need to initialize.")