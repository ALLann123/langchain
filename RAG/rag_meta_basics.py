#!/usr/bin/python3
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persist_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persist_directory}")

# Check if the chroma vector store already exists
if not os.path.exists(persist_directory):
    print("[+] Persistent directory does not exist. Initializing Vector store....")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory: {books_dir} does not exist. Please check the path"
        )

    # List all text files in the directory
    books_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []

    for book_file in books_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document chunk information ---")
    print(f"Number of document chunks: {len(docs)}")
    if docs:
        print(f"\nSample metadata: {docs[0].metadata}")
        print(f"Sample content (first 100 chars): {docs[0].page_content[:100]}...")

    # Create embeddings
    print("\n=== Create Embeddings ===")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("--- Finished creating embeddings ---")

    # Create and persist vector store
    print("\n[+] Creating vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # No need for explicit persist() in newer versions - it's automatic
    print("Vector store created successfully. Persistence is automatic.")

else:
    print("Vector store already exists. No need to initialize.")