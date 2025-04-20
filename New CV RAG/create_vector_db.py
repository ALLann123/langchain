#!/usr/bin/python3
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Changed to PDF loader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer

# Define the directory containing files and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
resumes_dir = os.path.join(current_dir, "documents")  # Changed from "books" to "resumes"
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Resumes Directory: {resumes_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist! Initializing vector store...")

    # Ensure resumes directory exists
    if not os.path.exists(resumes_dir):
        raise FileNotFoundError(
            f"The directory {resumes_dir} does not exist!"
        )
    
    # List all PDF files in the directory
    resume_files = [f for f in os.listdir(resumes_dir) if f.endswith(".pdf")]  # Changed to PDF

    # Read the content from each file and store it with metadata
    documents = []

    for resume_file in resume_files:
        file_path = os.path.join(resumes_dir, resume_file)
        
        loader = PyPDFLoader(file_path)
        resume_docs = loader.load()
        for doc in resume_docs:
            # Add metadata including source filename and page number
            doc.metadata = {
                "source": resume_file,
                "page": doc.metadata.get("page", 0) + 1  # Human-readable page numbering
            }
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # Increased overlap for better context retention
        separator="\n"      # Split at line breaks for better readability
    )
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n-----Document Chunks Information------\n")
    print(f"Number of resume files processed: {len(resume_files)}")
    print(f"Number of document chunks created: {len(docs)}")
    if docs:
        print("\nSample chunk:")
        print(f"Source: {docs[0].metadata['source']} (Page {docs[0].metadata['page']})")
        print(docs[0].page_content[:200] + "...")  # Show first 200 chars of first chunk

    # Create embeddings
    print("\n-----Creating Embeddings----\n")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("[+] Finished creating embeddings")

    # Create the vector store and persist it
    print("\n----Creating Vector store----\n")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:num_threads": 1}  # For better stability
    )
    print("\n[+] Finished creating and persisting vector store")

else:
    print("[-] Vector Store Already Exists")



"""
ew CV RAG> python .\create_vector_db.py
Resumes Directory: J:\code\AI\AI\New CV RAG\documents
Persistent directory: J:\code\AI\AI\New CV RAG\db\chroma_db_with_metadata
Persistent directory does not exist! Initializing vector store...        

-----Document Chunks Information------

Number of resume files processed: 3  
Number of document chunks created: 12

Sample chunk:
Source: allan.pdf (Page 1)
Allan Kariuki Mbugua                                                                                                     Contact: +254723269755
Github:https://github.com/ALLann123                    ...

-----Creating Embeddings----

[+] Finished creating embeddings

----Creating Vector store----


[+] Finished creating and persisting vector store
"""