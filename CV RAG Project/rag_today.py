#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import textwrap

load_dotenv()

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

model = {
    "model": MODEL,
    "messages": messages,
    "temperature": 0.7
}

"""Step 1:Load the documents"""
#define the directory containing the files
current_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(current_dir, "cvs")

#list to hold the documents
documents=[]

#get all the pdf files from the directory
file_paths=[os.path.join(pdfs_dir, file) for file in os.listdir(pdfs_dir) if file.endswith('.pdf')]

#load pdf using the pyPDFLoader
for file_path in file_paths:
    loader=UnstructuredMarkdownLoader(file_path)
    documents.extend(loader.load())

"""Step 2: Split the documents into chunks"""
#split the documents into chunks when storing into a vector database
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text_splitted_document=text_splitter.split_documents(documents)


"""step 3:Lets create embeddings and store in vector database"""
#create embedding
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
#lets create our FAISS vectore store
vectorstore=FAISS.from_documents(text_splitted_document, embeddings)

"""Step 4: Query Processing and Retrieving Data"""
query="user enters query here"

retriver=vectorstore.as_retriver(
    search_type="similarity",
    search_kwargs={"k":3},
)

result=retriver.invoke(query)

llm=model(temperature=0)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriver=retriver,
    return_source_documents=True
)

query="What is the names of the candidates? "
result=qa_chain({"query": query})
print("\nQuestion: ", query)

wrapped_text=textwrap.fill(result["result"], width=80)
print("Answer: \n", wrapped_text)