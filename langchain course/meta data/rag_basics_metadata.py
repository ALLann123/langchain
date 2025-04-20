#!/usr/bin/python3
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer


#define the directory containing files and persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
books_dir=os.path.join(current_dir, "books")
db_dir=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books Directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

#check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist!!Initializing vector store...")

    #ensure book directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist!!"
        )
    
    #list all the test files in  the directory
    book_files = [f for f in os.listdir(books_dir)if f.endswith(".txt")]

    #read the content from each file and store it with metadata
    documents=[]

    for book_file in book_files:
        file_path=os.path.join(books_dir, book_file)
        loader=TextLoader(file_path)
        books_doc=loader.load()
        for doc in books_doc:
            #add metadata to each document indicating its source
            doc.metadata={"source": book_file}
            documents.append(doc)

        #split the documents into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs=text_splitter.split_documents(documents)

    #display information about  the split documents
    print("\n-----Documents Chunks Information------\n")
    print(f"Number of document chunks: {len(docs)}")

    #create embeddings
    print("\n-----Creating Embeddings----\n")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Specify device
        encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
    )
    print("[+] Finished creating embeddings...\n")

    #create the vector store and persist it
    print("\n----Creating Vector store----\n")
    db=Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n[+]Finished Creating and persisting vector store...")

else:
    print("[-]Vector Store Already Exists!!")

    