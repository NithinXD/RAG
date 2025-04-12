import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from functools import lru_cache

load_dotenv()

# Configure API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use LRU cache to avoid redundant embedding calls for the same text
@lru_cache(maxsize=1000)
def get_embedding(text):
    """
    Get embedding for text using Gemini's embedding model.
    """
    response = genai.embed_content(model='models/embedding-001', content=text)
    return response['embedding']

def get_embedding_batch(texts, batch_size=5):
    """
    Get embeddings for a batch of texts efficiently.
    """
    results = []

    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = [get_embedding(text) for text in batch]
        results.extend(batch_embeddings)

    return results

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)