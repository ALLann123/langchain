import os
import PyPDF2
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict

# Configuration
CV_DIRECTORY = "./cvs"  # Path to your CVs directory
CHROMA_DB_PATH = "./chroma_db"  # Path to store ChromaDB
COLLECTION_NAME = "cv_collection"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Use sentence-transformers embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
    return text.strip()

def process_cvs(cv_directory: str) -> List[Dict]:
    """Process all CVs in the directory and return structured data."""
    cv_data = []
    for filename in os.listdir(cv_directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(cv_directory, filename)
            text = extract_text_from_pdf(filepath)
            if text:  # Only add if text was extracted successfully
                cv_data.append({
                    "filename": filename,
                    "text": text,
                    "metadata": {"source": filename}
                })
    return cv_data

def create_chroma_collection(cv_data: List[Dict]):
    """Create or reset ChromaDB collection and add CV documents."""
    # Delete collection if it already exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass  # Collection didn't exist
    
    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity
    )
    
    # Prepare documents, ids and metadatas
    documents = []
    metadatas = []
    ids = []
    
    for idx, cv in enumerate(cv_data):
        documents.append(cv["text"])
        metadatas.append(cv["metadata"])
        ids.append(str(idx))
    
    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def main():
    # Process all CVs
    print("Processing CVs...")
    cv_data = process_cvs(CV_DIRECTORY)
    
    if not cv_data:
        print("No CVs found or could be processed.")
        return
    
    print(f"Processed {len(cv_data)} CVs")
    
    # Create ChromaDB collection
    print("Creating ChromaDB collection...")
    collection = create_chroma_collection(cv_data)
    
    # Verify
    print(f"Collection '{COLLECTION_NAME}' created with {collection.count()} items")
    print("Sample query test:")
    results = collection.query(
        query_texts=["computer science degree"],
        n_results=2
    )
    print(results)

if __name__ == "__main__":
    main()