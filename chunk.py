from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from pprint import pprint

def get_token_nodes(documents: list[Document]):
    character_splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=0,
        paragraph_separator="\n\n",
    )

    token_splitter = TokenTextSplitter(
        chunk_size=256,     
        chunk_overlap=20,   
    )

    nodes = character_splitter.get_nodes_from_documents(documents)
    token_nodes = token_splitter.get_nodes_from_documents(nodes) 

    return token_nodes

