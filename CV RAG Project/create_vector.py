#!/usr/bin/python3
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def extract_candidate_info(text, filename):
    """Enhanced information extraction from CV text"""
    info = {"source": filename}
    
    # Name extraction with multiple patterns
    name_patterns = [
        r"Name:\s*(.+)",
        r"Resume\s*of\s*(.+)",
        r"Personal\s*Details[\s\S]*?Name[^a-zA-Z0-9]*([a-zA-Z ]+)",
        r"^([A-Z][a-z]+ [A-Z][a-z]+)$"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info["candidate_name"] = match.group(1).strip()
            break
    
    # Extract skills section
    skills_match = re.search(r"Skills:([\s\S]+?)(?=\n\n|\Z)", text, re.IGNORECASE)
    if skills_match:
        info["skills"] = skills_match.group(1).strip()
    
    # Extract certifications section
    certs_match = re.search(r"Certifications:([\s\S]+?)(?=\n\n|\Z)", text, re.IGNORECASE)
    if certs_match:
        info["certifications"] = certs_match.group(1).strip()
    
    return info

current_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(current_dir, "cvs")

documents = []

for file_path in [os.path.join(pdfs_dir, f) for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]:
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        
        # Extract and add metadata
        info = extract_candidate_info(full_text, os.path.basename(file_path))
        for page in pages:
            page.metadata.update(info)
            # Add candidate name to page content for better retrieval
            if "candidate_name" in info:
                page.page_content = f"Candidate: {info['candidate_name']}\n{page.page_content}"
        
        documents.extend(pages)
        print(f"Processed {os.path.basename(file_path)} - Name: {info.get('candidate_name', 'Not found')}")
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store created with enhanced metadata")