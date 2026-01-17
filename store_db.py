import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def get_or_create_index():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Saves to disk
    chroma_collection = chroma_client.get_or_create_collection("rag_knowledge_base", embedding_function=embedding_function)

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