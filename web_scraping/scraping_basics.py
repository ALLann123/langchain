#!/usr/bin/python3
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

#define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_kilimall")

#Step 1: Scrape the content off the site using webBaseLoader
#WebBaseLoader loads web pages and extracts their content
urls = ["https://www.kilimall.co.ke"]

#create a loader for the web content
loader = WebBaseLoader(urls)
document = loader.load()

#Step 2: Split the scraped content into chunks
#CharacterTextSplitter splits the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)  # Added overlap
docs = text_splitter.split_documents(document)

#display information about the split documents
print("\n--- Document Chunks Information ----")
print(f"Number of document Chunks: {len(docs)}")
print(f"Sample chunk: \n{docs[0].page_content[:200]}...\n")  # Show first 200 chars

#step 3: Create embeddings for the document chunks
#Use Huggingface free embedding to convert text to its numerical vectors
embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': False}  # Changed to False
)

#step 4: Create a persistent vector store with the embeddings
#Chroma stores the embeddings for effiecient searching
if not os.path.exists(persistent_directory):
    print("**********************************************************************")
    print(f"Creating Vector store in {persistent_directory}")
    db = Chroma.from_documents(docs, embeddings, persist_directory = persistent_directory)
    print(f"[+] Finished Creating vector store in {persistent_directory}")

# step 5: Query the vector store
#create a retriver for querying the vector store
retriever = db.as_retriever(
    search_type = "similarity",  # Changed from similarity_score_threshold
    search_kwargs = {
        "k": 3  # Removed score_threshold
    }
)

#define the users question
query = "What products does Kilimall sell?"

#retrive relevant documents based on query
relevant_docs = retriever.get_relevant_documents(query)  # Changed invoke to get_relevant_documents

#display relevant results
print("\n--- Relevant Documents ---")
print()
print(f"Question asked: {query}")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content[:500]}...\n")  # Show first 500 chars
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")