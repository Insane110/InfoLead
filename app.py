import streamlit as st
import time
from crawl_fastapi import crawl_via_fastapi
from fetch_urls import fetch_top_n_links
from docs_parse import parse_document
from chunk import get_token_nodes
from store_db import get_or_create_index, get_or_create_doc_index 
from query_cache import is_similar_query_cached, save_query_to_cache, clear_query_cache
from fetch_llm import output_llm, REFINEMENT_PROMPT
import os
import tempfile
from rank_bm25 import BM25Okapi
# from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from collections import defaultdict
import shutil

st.set_page_config(
    page_title="InfoLead",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        font-size: 1.1rem;
    }
    
    h1, h2, h3 {
        font-size: 2.2rem !important;
    }
    
    .stMarkdown h2 {
        font-size: 2.2rem !important;
    }
    
    .stMarkdown h3 {
        font-size: 1.6rem !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        font-size: 1.2rem !important;
    }
    
    label {
        font-size: 1.15rem !important;
    }
    
    .stTextInput input, .stTextArea textarea {
        font-size: 1.1rem !important;
    }
    
    .stButton button {
        font-size: 1.15rem !important;
        padding: 0.7rem 1.3rem !important;
    }
    
    .stCheckbox label {
        font-size: 1.15rem !important;
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 1.15rem !important;
    }
    
    .caption, small {
        font-size: 1rem !important;
    }
    
    [data-testid="stFileUploader"] {
        font-size: 1.1rem !important;
    }
    
    .stSlider label {
        font-size: 1.15rem !important;
    }
    
    [data-testid="stSidebar"] {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] label {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        font-size: 1.2rem !important;
        padding: 0.8rem 1.4rem !important;
    }
    
    [data-testid="stSidebar"] input {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        font-size: 1.15rem !important;
    }
</style>
""", unsafe_allow_html=True)

def RRF(vector_nodes, bm25_nodes, k=60):
    scores = defaultdict(float)
    for rank, node_with_score in enumerate(vector_nodes):
        node_id = node_with_score.node.node_id
        scores[node_id] += 1 / (k + rank + 1)
    for rank, node_with_score in enumerate(bm25_nodes):
        node_id = node_with_score.node.node_id
        scores[node_id] += 1 / (k + rank + 1)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_nodes = []
    for node_id, score in sorted_nodes:
        node = next(n.node for n in vector_nodes + bm25_nodes if n.node.node_id == node_id)
        fused_nodes.append(NodeWithScore(node=node, score=score))
    return fused_nodes

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        return RRF(vector_nodes, bm25_nodes)

class SimpleBM25Retriever(BaseRetriever):
    def __init__(self, nodes, similarity_top_k=5):
        super().__init__()
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        corpus = [node.text.split() for node in nodes]
        self.bm25 = BM25Okapi(corpus)

    def _retrieve(self, query_bundle: QueryBundle):
        query_tokens = query_bundle.query_str.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.nodes, scores), key=lambda x: x[1], reverse=True)[:self.similarity_top_k]
        return [NodeWithScore(node=node, score=float(score)) for node, score in ranked]
    
@st.cache_resource
def load_index():
    return get_or_create_index()

@st.cache_resource
def load_reranker():
    return SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=2)

if "llm" not in st.session_state:
    st.session_state.llm = output_llm()

if "index" not in st.session_state:
    st.session_state.index, st.session_state.embed_model = load_index()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "use_history" not in st.session_state:
    st.session_state.use_history = True

if "history_length" not in st.session_state:
    st.session_state.history_length = 3

if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = []

if "doc_nodes" not in st.session_state:
    st.session_state.doc_nodes = None

if "doc_inserted" not in st.session_state:
    st.session_state.doc_inserted = False

if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None

if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

def process_query(user_query, search_internet, search_query, num_results, attach_doc, file_path, chat_history):
    web_docs = []

    if search_internet and search_query:
        already_cached, matched_query = is_similar_query_cached(search_query)
        if already_cached:
            st.info(f"💾 Similar query found in knowledge base: *'{matched_query}'* — skipping web scrape")
        else:
            with st.status("🌐 Searching the internet...", expanded=True):
                top_links = fetch_top_n_links(search_query, num_results)
                web_docs = crawl_via_fastapi(top_links)
                save_query_to_cache(search_query)

    doc_docs = []
    nodes = None
    if attach_doc and file_path:
        if not st.session_state.doc_inserted:
            doc_index, nodes, cache_hit = get_or_create_doc_index(file_path)
            label = "📦 Loading cached index (no re-embedding)..." if cache_hit else "📄 Parsing & embedding document (first time)..."
            with st.status(label, expanded=True):
                st.session_state.index = doc_index
                st.session_state.doc_nodes = nodes
                st.session_state.parsed_docs = nodes 
                st.session_state.bm25_retriever = SimpleBM25Retriever(nodes=nodes, similarity_top_k=5)
                st.session_state.doc_inserted = True
                st.session_state.docs_indexed = True
                st.session_state["engine_nodes_changed"] = True

            doc_docs = nodes

        else:
            st.info("📄 Using previously parsed document")
            doc_docs = st.session_state.parsed_docs
            nodes = st.session_state.doc_nodes

    elif web_docs:
        with st.status("📚 Updating knowledge base...", expanded=True):
            nodes = get_token_nodes(web_docs)
            st.session_state.index.insert_nodes(nodes)
            st.session_state.bm25_retriever = SimpleBM25Retriever(nodes=nodes, similarity_top_k=5)
            # st.session_state.index.storage_context.persist()

    context_query = user_query
    use_history = st.session_state.get('use_history', True)
    history_length = st.session_state.get('history_length', 3)

    if chat_history and use_history:
        recent_history = chat_history[-history_length:]
        history_context = "\n\n".join([
            f"Previous Question: {chat['query']}\nPrevious Answer: {chat['response']}"
            for chat in recent_history
        ])
        context_query = f"""Conversation History:
    {history_context}

    Current Question: {user_query}

    Please answer the current question, taking into account the conversation history above for context."""

    if nodes:
        vector_retriever = VectorIndexRetriever(index=st.session_state.index, similarity_top_k=2)
        bm25_retriever = st.session_state.bm25_retriever
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        engine_key = "query_engine_hybrid"
        if engine_key not in st.session_state or st.session_state.get("engine_nodes_changed"):
            qa_prompt = PromptTemplate(REFINEMENT_PROMPT)
            response_synthesizer = get_response_synthesizer(
                llm=st.session_state.llm,
                text_qa_template=qa_prompt,
                streaming=True             
            )
            st.session_state[engine_key] = RetrieverQueryEngine(
                retriever=hybrid_retriever,
                node_postprocessors=[],     # reranker removed for streaming stability
                response_synthesizer=response_synthesizer
            )
            st.session_state["engine_nodes_changed"] = False

        query_engine = st.session_state[engine_key]
    else:
        query_engine = st.session_state.index.as_query_engine(
            llm=st.session_state.llm,
            embed_model=st.session_state.embed_model,
            streaming=True                  
        )

    response = query_engine.query(context_query)
    return response.response_gen, len(web_docs), len(doc_docs)  

def main():
    st.markdown("## 🔍 InfoLead")
    st.markdown("Ask questions using web search and your documents")
    st.divider()

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        search_internet = st.checkbox("Search Internet", value=False)
        search_query = ""
        num_results = 5

        if search_internet:
            search_query = st.text_input("Search Query", placeholder="Leave empty to use main query")
            num_results = st.slider("Number of Results", 1, 10, 5)

        st.divider()

        attach_doc = st.checkbox("Attach Document", value=False)
        file_path = None

        if attach_doc:
            if st.session_state.current_file_path and st.session_state.docs_indexed:
                st.info(f"✅ Document indexed: {os.path.basename(st.session_state.current_file_path)}")
            
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "md"])
            if uploaded_file and not st.session_state.current_file_path:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    st.session_state.current_file_path = tmp.name
                    file_path = tmp.name
                
                st.success(f"📄 New file ready: {uploaded_file.name}")
            if st.session_state.current_file_path:
                file_path = st.session_state.current_file_path

        st.divider()

        st.markdown("### 💭 Conversation Settings")
        use_history = st.checkbox("Use Conversation History", value=True, 
                                   help="Include previous messages for context in follow-up questions")
        history_length = 3
        if use_history:
            history_length = st.slider("Messages to Remember", 1, 10, 3, 
                                       help="Number of previous Q&A pairs to include")
        
        st.session_state.use_history = use_history
        st.session_state.history_length = history_length

        st.divider()

        if st.button("🗑️ Clear History", use_container_width=True, key="clear_doc_btn"):
            clear_query_cache()
            st.session_state.chat_history = []
            st.session_state.parsed_docs = []
            st.session_state.doc_nodes = None
            st.session_state.doc_inserted = False
            st.session_state.current_file_path = None
            st.session_state.docs_indexed = False
            st.session_state.bm25_retriever = None
            if os.path.exists("storage_rag_knowledge_base"):
                shutil.rmtree("storage_rag_knowledge_base")

            st.session_state.index, st.session_state.embed_model = load_index()
            st.success("All data cleared!")
            st.rerun()

    st.subheader("💬 Chat")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.write(chat["response"])
            c1, c2, c3 = st.columns(3)
            c1.caption(f"⏱️ {chat['time']:.2f}s")
            if chat["web_docs"]:
                c2.caption(f"🌐 Web docs: {chat['web_docs']}")
            if chat["doc_docs"]:
                c3.caption(f"📄 File docs: {chat['doc_docs']}")

    user_query = st.chat_input("Ask something...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        start = time.time()
        with st.chat_message("assistant"):
            final_search_query = search_query or user_query
            response_gen, web_docs, doc_docs = process_query(
                user_query, search_internet, final_search_query,
                num_results, attach_doc, file_path, st.session_state.chat_history)

            full_response = ""
            placeholder = st.empty()
            for token in response_gen:
                full_response += token
                placeholder.markdown(full_response + "▌")  
            placeholder.markdown(full_response)           

            elapsed = time.time() - start
            c1, c2, c3 = st.columns(3)
            c1.caption(f"⏱️ {elapsed:.2f}s")
            if web_docs:
                c2.caption(f"🌐 Web docs: {web_docs}")
            if doc_docs:
                c3.caption(f"📄 File docs: {doc_docs}")

        st.session_state.chat_history.append({
            "query": user_query,
            "response": full_response, 
            "time": elapsed,
            "web_docs": web_docs,
            "doc_docs": doc_docs
        })

        st.rerun()

if __name__ == "__main__":
    main()
