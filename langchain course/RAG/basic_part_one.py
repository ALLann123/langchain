#!/usr/bin/python3
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer

#define the directory containing the text file and the persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir, "document", "mrrobot.txt")
persistent_directory=os.path.join(current_dir, "db", "chroma_db")

#check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("[-]No Vector Store. Initializing vector store....")

    #ensure the text file exits
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exit. Please check the path."

        )
    
    #read the text content from the file
    loader=TextLoader(file_path)
    documents=loader.load()


    #split the document into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs=text_splitter.split_documents(documents)

    #Display information about the split documents
    print("\n---DOcument Chuncks Information---\n")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample Chunk: \n{docs[0].page_content}\n")

    #create embeddings
    print("\n-----Creating Embeddings----\n")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Specify device
        encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
    )
    print("[+] Finished creating embeddings...\n")

    #Create the vecor store and persist it automatically
    print("\n----Creating Vector store----\n")
    db=Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )

else:
    print("Vector Store Already exists. No need to intialize")




