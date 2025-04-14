#!/usr/bin/python3
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "faiss_db")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS
db = FAISS.load_local(persistent_directory, embeddings, allow_dangerous_deserialization=True)

# Query
query = "Who had gone off to the Ethiopians?"

# First try direct similarity search
print("----[+] Top Matching Documents ----")
results = db.similarity_search_with_relevance_scores(query, k=3)
for i, (doc, score) in enumerate(results, 1):
    print(f"Document {i} (Score: {score:.3f}):")
    print(doc.page_content)
    if hasattr(doc, 'metadata'):
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 50)

# Then try with retriever (lower threshold)
print("\n----[+] Using Retriever ----")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1}  # Much lower threshold
)
relevant_docs = retriever.invoke(query)

if relevant_docs:
    print("Relevant Documents Found:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
else:
    print("No documents met threshold. Showing top match anyway:")
    print(results[0][0].page_content)