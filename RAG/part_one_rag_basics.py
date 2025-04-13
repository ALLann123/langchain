#!/usr/bin/python3
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import os

#define the directory containing the file and the persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "faiss_db")
#the above is where we retrieve the pdf/txt file we want to use


#now check if the vector store i.e chroma db already exists
if not os.path.exists(persistent_directory):
    print("[+]Persistent directory does not exist. Initializing vector store....")
    #now lets check if the file exists and if not raise an error
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    #read the text content of the file
    loader=TextLoader(file_path)
    documents=loader.load()

    #now lets split the document into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs=text_splitter.split_documents(documents)

    #display information about the split documents
    print("\n--- Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n {docs[0].page_content}\n")

    #create embeddings i.e numerical representation of text
    print("===================================================")
    print("\n--- Creating Embeddings---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persistent_directory)

    print("[+] FAISS vector store created and saved.")

else:
    print("[-]Vector store already exists. No need to initialize!!")

