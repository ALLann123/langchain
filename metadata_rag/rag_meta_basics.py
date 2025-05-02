#!/usr/bin/python3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory containing the PDF files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(current_dir, "cvs")
db_dir = os.path.join(current_dir, "db")
persist_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"PDF directory: {pdfs_dir}")
print(f"Persistent directory: {persist_directory}")

# Check if the chroma vector store already exists
if not os.path.exists(persist_directory):
    print("[+] Persistent directory does not exist. Initializing Vector store....")

    # Ensure the PDF directory exists
    if not os.path.exists(pdfs_dir):
        raise FileNotFoundError(
            f"The directory: {pdfs_dir} does not exist. Please check the path"
        )

    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdfs_dir}")

    # Read the content from each PDF file and store it with metadata
    documents = []

    for pdf_file in pdf_files:
        file_path = os.path.join(pdfs_dir, pdf_file)
        try:
            print(f"Loading {pdf_file}...")
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                # Add metadata to each document indicating its source
                doc.metadata = {"source": pdf_file}
                documents.append(doc)
            print(f"Successfully loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
            continue

    if not documents:
        raise ValueError("No valid documents were loaded from the PDF files")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] #Added more separators.
    )
    docs = text_splitter.split_documents(documents)

    # Check for empty documents after splitting
    docs = [doc for doc in docs if len(doc.page_content.strip()) > 0]

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
    print("Vector store created successfully. Persistence is automatic.")

else:
    print("Vector store already exists. No need to initialize.")