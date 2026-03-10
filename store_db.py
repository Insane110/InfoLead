import chromadb
import hashlib
import json
from pathlib import Path
from docs_parse import parse_document
from chunk import get_token_nodes
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

def get_or_create_index():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    try:
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    except:
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
    index.storage_context.persist(persist_dir="./storage_rag_knowledge_base")
    return index, embed_model

def _file_fingerprint(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def _index_key(file_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "sha256": _file_fingerprint(file_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model_name,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:20]

def get_or_create_doc_index(file_path: str, chunk_size=256, chunk_overlap=20):
    """
    Returns (index, nodes, cache_hit).
    Cache hit  = same file + same params seen before → loads from disk, no re-embedding.
    Cache miss = new file → parses, chunks, embeds, saves to disk.
    """
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    key = _index_key(file_path, chunk_size, chunk_overlap, embed_model_name)
    persist_dir = f"./storage_rag_knowledge_base/{key}"
    meta_file   = Path(persist_dir) / "meta.json"

    embed_model    = HuggingFaceEmbedding(model_name=embed_model_name)
    chroma_client  = chromadb.PersistentClient(path="./chroma_db")          
    chroma_collection = chroma_client.get_or_create_collection(f"doc_{key}") 
    vector_store   = ChromaVectorStore(chroma_collection=chroma_collection)

    if Path(persist_dir).exists() and meta_file.exists():
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir       
        )
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        nodes_data = json.loads((Path(persist_dir) / "nodes.json").read_text())
        from llama_index.core.schema import TextNode
        nodes = [TextNode.parse_raw(n) for n in nodes_data]
        return index, nodes, True

    parsed = parse_document(file_path)
    nodes  = get_token_nodes(parsed)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    storage_context.persist(persist_dir=persist_dir)   
    meta_file.write_text(json.dumps({
        "file_path":     str(file_path),
        "chunk_size":    chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model":   embed_model_name,
    }, indent=2))
    (Path(persist_dir) / "nodes.json").write_text(
        json.dumps([n.json() for n in nodes])
    )

    return index, nodes, False
