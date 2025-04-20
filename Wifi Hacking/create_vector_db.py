#!/usr/bin/python3
import os
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer

# Pre-download the model to avoid hanging
def ensure_model_downloaded():
    print("[-] Checking/Downloading HuggingFace model...")
    SentenceTransformer("all-MiniLM-L6-v2")
    print("[+] Model is ready")

#define the directory containing the PDF file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "book", "Wireless_Pwn.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

#check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("[-] No Vector Store. Initializing vector store....")

    #ensure the PDF file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Pre-download model before starting the process
    ensure_model_downloaded()
    
    #read the PDF content
    print("[-] Loading PDF documents...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #split the document into chunks
    print("[-] Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #Display information about the split documents
    print("\n--- Document Chunks Information ---\n")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Number of pages in PDF: {len(documents)}")
    print(f"Sample Chunk (first 100 chars): \n{docs[0].page_content[:100]}...\n")

    #create embeddings
    print("\n----- Creating Embeddings -----\n")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("[+] Embeddings model loaded successfully")

    #Create the vector store with progress tracking
    print("\n---- Creating Vector store ----\n")
    db = Chroma.from_documents(
        tqdm(docs, desc="Processing documents"),
        embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:num_threads": 1}
    )
    print("[+] Vector store created and persisted successfully!")

else:
    print("[+] Vector Store already exists. No need to initialize")




"""
 python .\create_vector_db.py
[-] No Vector Store. Initializing vector store....
[-] Checking/Downloading HuggingFace model...
[+] Model is ready
[-] Loading PDF documents...
[-] Splitting documents...

--- Document Chunks Information ---

Number of document chunks: 13
Number of pages in PDF: 13
Sample Chunk (first 100 chars):
WPA/WPA2 Password Cracking by own_the_network 


Contents
Chapter One ..............................    


----- Creating Embeddings -----

[+] Embeddings model loaded successfully

---- Creating Vector store ----

Processing documents: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<?, ?it/s] 
[+] Vector store created and persisted successfully!
PS J:\code\AI\AI\Wifi Hacking> 
"""