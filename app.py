import streamlit as st
import time
from crawl_fastapi import crawl_via_fastapi
from fetch_urls import fetch_top_n_links
from docs_parse import parse_document
from chunk import get_token_nodes
from store_db import get_or_create_index
from fetch_llm import output_llm, REFINEMENT_PROMPT
import os
import tempfile

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

@st.cache_resource
def load_index():
    return get_or_create_index()

if "llm" not in st.session_state:
    st.session_state.llm = output_llm()

if "index" not in st.session_state:
    st.session_state.index, st.session_state.embed_model = load_index()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_query(user_query, search_internet, search_query, num_results, attach_doc, file_path):
    web_docs = []

    if search_internet and search_query:
        with st.status("🌐 Searching the internet...", expanded=True):
            top_links = fetch_top_n_links(search_query, num_results)
            web_docs = crawl_via_fastapi(top_links)

    doc_docs = []
    if attach_doc and file_path:
        with st.status("📄 Processing document...", expanded=True):
            doc_docs = parse_document(file_path)

    all_new_docs = web_docs + doc_docs
    if all_new_docs:
        with st.status("📚 Updating knowledge base...", expanded=True):
            nodes = get_token_nodes(all_new_docs)
            st.session_state.index.insert_nodes(nodes)
            st.session_state.index.storage_context.persist()

    with st.status("🤖 Generating response...", expanded=True):
        query_engine = st.session_state.index.as_query_engine(
            llm=st.session_state.llm,
            embed_model=st.session_state.embed_model
        )
        response = query_engine.query(user_query)
        prompt = REFINEMENT_PROMPT.format(raw_response=response.response)
        refined_response = st.session_state.llm.complete(prompt).text

    return refined_response, len(web_docs), len(doc_docs)

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
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "md"])
            if uploaded_file:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    file_path = tmp.name
                st.success(f"Loaded: {uploaded_file.name}")

        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
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
            response, web_docs, doc_docs = process_query(
                user_query, search_internet, final_search_query,
                num_results, attach_doc, file_path)
            elapsed = time.time() - start
            st.write(response)
            c1, c2, c3 = st.columns(3)
            c1.caption(f"⏱️ {elapsed:.2f}s")
            if web_docs:
                c2.caption(f"🌐 Web docs: {web_docs}")
            if doc_docs:
                c3.caption(f"📄 File docs: {doc_docs}")

        st.session_state.chat_history.append({
            "query": user_query,
            "response": response,
            "time": elapsed,
            "web_docs": web_docs,
            "doc_docs": doc_docs
        })

        st.rerun()

if __name__ == "__main__":
    main()
