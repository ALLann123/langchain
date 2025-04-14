#!/usr/bin/python3
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

def normalize_scores(docs_with_scores):
    """Invert and normalize scores since lower is better in this case"""
    scores = [score for _, score in docs_with_scores]
    max_score = max(scores)
    normalized = []
    
    for doc, score in docs_with_scores:
        # Invert so higher is better, then normalize
        inv_score = max_score - score
        normalized.append((doc, inv_score))
    
    # Now normalize to 0-1 range
    max_inv = max(score for _, score in normalized)
    if max_inv == 0:  # All scores equal
        return [(doc, 1.0) for doc, _ in normalized]
    
    return [
        (doc, score / max_inv)
        for doc, score in normalized
    ]

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load embeddings with proper settings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}  # Chroma handles normalization
)

# Load Chroma
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Query
query = "Who had gone off to the Ethiopians?"

# Get results
print("----[+] Search Results ----")
results = db.similarity_search_with_score(query, k=3)

# Process scores (Chroma returns L2 distance where lower=better)
print("\nRaw Scores (lower is better):")
for i, (doc, score) in enumerate(results, 1):
    print(f"Document {i}: {score:.3f}")

# Normalize with proper inversion
normalized = normalize_scores(results)

print("\n----[+] Normalized Scores (higher is better) ----")
for i, (doc, score) in enumerate(normalized, 1):
    print(f"Document {i}: {score:.3f}")

# Best match is now the one with highest normalized score
best_match, best_score = max(normalized, key=lambda x: x[1])

print("\n----[+] Best Match ----")
print(best_match.page_content)
if hasattr(best_match, 'metadata'):
    print(f"\nSource: {best_match.metadata.get('source', 'Unknown')}")