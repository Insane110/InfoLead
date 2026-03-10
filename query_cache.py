import json
import numpy as np
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

CACHE_FILE = "./storage_rag_knowledge_base/searched_queries.json"
SIMILARITY_THRESHOLD = 0.85

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _load_cache():
    p = Path(CACHE_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return []  

def _save_cache(cache):
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_FILE).write_text(json.dumps(cache))

def is_similar_query_cached(new_query: str):
    """
    Returns (True, matched_query) if a similar query was already searched.
    Returns (False, None) if this is a new query.
    """
    cache = _load_cache()
    if not cache:
        return False, None

    embed_model = _get_embed_model()
    new_embedding = embed_model.get_text_embedding(new_query)

    for entry in cache:
        sim = _cosine_similarity(new_embedding, entry["embedding"])
        if sim >= SIMILARITY_THRESHOLD:
            return True, entry["query"]

    return False, None

def save_query_to_cache(query: str):
    """Save a newly searched query + its embedding to cache."""
    cache = _load_cache()
    embed_model = _get_embed_model()
    embedding = embed_model.get_text_embedding(query)
    cache.append({"query": query, "embedding": embedding})
    _save_cache(cache)

def clear_query_cache():
    Path(CACHE_FILE).unlink(missing_ok=True)