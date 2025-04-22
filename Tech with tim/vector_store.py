#!/usr/bin/python3
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
import os
import pandas as pd
from langchain_core.documents import Document

# we are using this to read our data from our file into memory
df = pd.read_csv("realistic_restaurant_reviews.csv")

# lets bring in the embedding models
print("\n-----Creating Embeddings----\n")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# where we store our db
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)  # fixed typo: exits -> exists

if add_documents:
    # create a list
    documents = []
    ids = []
    # the for loop below goes row by row
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},  # fixed syntax in metadata
            id=str(i)
        )
        ids.append(str(i))  # indentation fixed to be inside loop
        documents.append(document)

# create the vector store
vector_store = Chroma(
    collection_name="restaurant_review",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print("[+]Vector Store created")
    print()
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
